"""Rival-anchor verification (RAV) - decision-time synthesized impostors.

Replaces the phone-sequence verifier as the pipeline's precision stage.
Instead of consulting a separate phone recognizer, the system synthesizes its
own impostors at enrollment time: the keyword's phonetic near-neighbours are
derived from its text (CMUdict / g2p_en phone sequences, edit distance), each
rival is rendered through the same TTS bank as the anchor, and embedded into
per-rival centroids with the same embedding function the detector uses. A
detection is kept only if its window embedding beats EVERY rival by a
calibrated margin:

    margin(w) = cos(w, anchor) - max_r cos(w, rival_r)  >=  delta

The comparison is differential (immune to absolute score drift), runs in the
exact embedding space the attentive head was trained to make discriminative
for confusable pairs (its training batches were half phonetic near-neighbours),
and costs a handful of dot products - no second model, no CTC decode.

Design notes, each forced by a known failure mode of the retired phone stage:
  * Perfect homophones (edit distance 0, e.g. capital/capitol) are EXCLUDED
    from the rival list - an identical-phone rival would tie every margin at
    zero and reject the keyword itself. Homophones remain out of scope for
    any acoustic verifier.
  * The keyword's own stem family is excluded ("parties" must not rival
    "party"), mirroring the calibration guard's word-family rule.
  * delta is calibrated from synthetic material only: the anchor's held-out
    positive voices must clear it, the rival clips themselves must not. No
    stream audio, no transcripts, deterministic.
  * Verification re-embeds windows through the detector's own chunk-context
    pooling path (frames() + pool_windows()), never isolated forward passes -
    the protocol-alignment lesson from calibration applies here too.

Contamination alarm: a supposedly keyword-free calibration window whose
rival margin is strongly positive (it "is" the keyword) indicates transcript
leakage; `margin_alarm()` exposes that check.
"""

import os
import re

import numpy as np

from scoring import (SAMPLE_RATE, artifact_suffix, l2_normalize,
                     _keyword_prefix)

N_RIVALS = 10
N_PHONE_RIVALS = 8           # lexicon stage: form-close words, any frequency
N_BANK_RIVALS = 8            # bank stage: embedding-close frequent words
CLUSTER_GAP_S = 0.3          # detection-event clustering, same as phone stage
TOP_WINDOWS_PER_EVENT = 5    # windows whose margin median decides an event
MIN_RIVAL_GAP = 0.15         # a rival's own clips must lose by at least this
MIN_POS_GAP = 0.30           # keyword positives must beat each rival by this
DELTA_FLOOR = 0.0            # never accept a detection that loses the argmax


def bank_path():
    """Global rival bank (model-level artifact, lives with the code repo
    regardless of SIAMESE_PROJECT_ROOT; override with SIAMESE_RIVAL_BANK)."""
    env = os.environ.get("SIAMESE_RIVAL_BANK")
    if env:
        return env
    code_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(code_root, "keywords",
                        f"rival_bank{artifact_suffix()}.npz")


# --------------------------------------------------------------------------
# Rival word selection (text -> phonetic near-neighbours)
# --------------------------------------------------------------------------

def _strip_stress(phones):
    return tuple(re.sub(r"\d", "", p) for p in phones)


def _edit_distance(a, b):
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j - 1] + cost, prev[j] + 1, cur[j - 1] + 1)
        prev = cur
    return prev[m]


def keyword_phones(keyword):
    """CMUdict lookup with g2p_en fallback (both offline)."""
    kw = keyword.lower().strip()
    try:
        from nltk.corpus import cmudict
        entry = cmudict.dict().get(kw)
        if entry:
            return _strip_stress(entry[0])
    except Exception:
        pass
    from g2p_en import G2p
    return _strip_stress([p for p in G2p()(kw) if p.strip() and p != " "])


SCHWA_FAMILY = {"AH", "ER", "IH", "AX", "UH"}   # reduced-vowel phones


def _effective_homophone(kw_ph, ph):
    """True for pairs a verifier cannot separate in connected speech.

    Identical phone strings are homophones outright. A single edit whose
    phones are all reduced-vowel class (schwa family) is an *effective*
    homophone: casual speech reduces or drops those vowels, so the two words
    collapse acoustically (ireland /AY ER L AH N D/ vs island /AY L AH N D/
    differ only by the unstressed ER - broadcast pronunciation of the first
    IS the second). Arming such a rival rejects the keyword itself.
    """
    d = _edit_distance(kw_ph, ph)
    if d == 0:
        return True
    if d == 1:
        # locate the single edit via LCS-style alignment on the two ends
        i = 0
        while i < min(len(kw_ph), len(ph)) and kw_ph[i] == ph[i]:
            i += 1
        j = 0
        while (j < min(len(kw_ph), len(ph)) - i
               and kw_ph[len(kw_ph) - 1 - j] == ph[len(ph) - 1 - j]):
            j += 1
        edited = set(kw_ph[i:len(kw_ph) - j]) | set(ph[i:len(ph) - j])
        if edited and edited <= SCHWA_FAMILY:
            return True
    return False


MAX_PHONE_OVERLAP = 0.7      # contiguous-overlap ratio above which a rival
                             # is a partial rendering of the keyword itself


def _norm_reduced(seq):
    """Collapse reduced-vowel phones to one token: dictionary pronunciations
    disagree on which schwa-family vowel sits in unstressed positions
    (adminIHstration vs minAHstration), and speech reduces them identically."""
    return tuple("AX" if p in SCHWA_FAMILY else p for p in seq)


def _contig_overlap(a, b):
    """Longest common contiguous phone substring / min length,
    schwa-normalized."""
    a, b = _norm_reduced(a), _norm_reduced(b)
    best = 0
    for i in range(len(a)):
        for j in range(len(b)):
            k = 0
            while (i + k < len(a) and j + k < len(b)
                   and a[i + k] == b[j + k]):
                k += 1
            best = max(best, k)
    return best / max(1, min(len(a), len(b)))


def _excluded(word, kw, stem, kw_ph, log):
    """Family members, (effective) homophones, and phone-substring words are
    never rivals.

    The substring rule exists because the detector's multi-scale windows
    legitimately cover PARTIAL words: a 0.6-scale window on a genuine
    "administration" is acoustically "ministration", and on a genuine
    "party" it is "part" - arming such a rival rejects the keyword's own
    partial windows.
    """
    if word == kw or not word.isalpha():
        return True
    if stem and word.startswith(stem):
        return True                       # keyword family - keep matching
    try:
        from nltk.corpus import cmudict
        prons = cmudict.dict().get(word)
        ph = _strip_stress(prons[0]) if prons else None
    except Exception:
        ph = None
    if ph is None:
        return False
    if _effective_homophone(kw_ph, ph):
        log(f"  '{word}' excluded from rivals (effective homophone - "
            f"out of scope for acoustic separation)")
        return True
    if _contig_overlap(kw_ph, ph) >= MAX_PHONE_OVERLAP:
        log(f"  '{word}' excluded from rivals (phone-substring of the "
            f"keyword - would match the keyword's own partial windows)")
        return True
    return False


def find_rivals(keyword, anchor=None, top_n=N_RIVALS, log=print):
    """Two-stage rival selection.

    Stage 1 (lexicon): nearest CMUdict words by phone edit distance - catches
    form-close words regardless of frequency ("island" for ireland,
    "immigration" for administration). Stage 2 (bank, when the global rival
    bank exists and an anchor is given): frequent words ranked by embedding
    proximity to the anchor - the detector's OWN nearest confusables, which
    phone distance can miss entirely ("policy" ranks 845th in phone space for
    party but 6th in embedding space). The union, family- and
    homophone-filtered, is the rival set.
    """
    from nltk.corpus import cmudict
    kw = keyword.lower().strip()
    kw_ph = keyword_phones(kw)
    stem = _keyword_prefix(kw)

    # stage 1 - lexicon / phone space
    scored = []
    for word, prons in cmudict.dict().items():
        if word == kw or not word.isalpha():
            continue
        if stem and word.startswith(stem):
            continue
        ph = _strip_stress(prons[0])
        if abs(len(ph) - len(kw_ph)) > 3:
            continue
        d = _edit_distance(kw_ph, ph)
        if (_effective_homophone(kw_ph, ph)
                or _contig_overlap(kw_ph, ph) >= MAX_PHONE_OVERLAP):
            continue                      # homophone / substring - skip
        common = 0
        for x, y in zip(kw_ph, ph):
            if x != y:
                break
            common += 1
        scored.append((d / max(len(kw_ph), len(ph)), -common, word))
    scored.sort()
    rivals = [w for _, _, w in scored[:N_PHONE_RIVALS]]
    log(f"  lexicon rivals: {', '.join(rivals)}")

    # stage 2 - embedding space over the global bank
    bp = bank_path()
    if anchor is not None and os.path.exists(bp):
        z = np.load(bp, allow_pickle=False)
        words, embs = [str(w) for w in z["words"]], z["embs"]
        sims = embs @ anchor
        picked = []
        for i in np.argsort(-sims):
            w = words[i]
            if w in rivals or _excluded(w, kw, stem, kw_ph, log):
                continue
            picked.append(w)
            if len(picked) >= N_BANK_RIVALS:
                break
        log(f"  embedding rivals (bank of {len(words)}): {', '.join(picked)}")
        rivals += picked
    elif anchor is not None:
        log("  (no rival bank on disk - lexicon stage only; build with "
            "pipeline/rival_bank.py)")
    return rivals[: max(top_n, N_PHONE_RIVALS + N_BANK_RIVALS)]


# --------------------------------------------------------------------------
# Margins and calibration
# --------------------------------------------------------------------------

def margins(window_embs, anchor, rival_centroids, cohort=None):
    """Keyword-vs-best-rival margin per window embedding.

    With a cohort, each anchor's scores are AS-normalized first (the same
    top-k standardization the detector uses), which removes per-anchor bias:
    the keyword anchor is built from ~30 augmented multi-voice clips while
    rival centroids come from 7 clean canonical clips, so their raw cosine
    scales are not comparable - the anchor-side normalization of AS-norm is
    exactly the correction for that. Without a cohort, falls back to raw
    cosine differences.
    """
    from scoring import asnorm_windows
    window_embs = np.atleast_2d(window_embs)
    if cohort is not None:
        s_kw = asnorm_windows(window_embs, anchor, cohort)[0]
        s_riv = np.stack([asnorm_windows(window_embs, r, cohort)[0]
                          for r in rival_centroids], axis=1).max(axis=1)
        return s_kw - s_riv
    s_kw = window_embs @ anchor
    s_riv = (window_embs @ rival_centroids.T).max(axis=1)
    return s_kw - s_riv


def calibrate_delta(pos_embs, rival_clip_embs, anchor, rival_centroids,
                    cohort=None, log=print):
    """Accept margin: midpoint between what impostor audio achieves and what
    the held-out enrollment positives achieve, floored at the argmax boundary.

    Both sides are synthetic (holdout TTS voices vs. the rival clips
    themselves), so calibration needs no stream audio and is deterministic.
    """
    pos_m = margins(pos_embs, anchor, rival_centroids, cohort)
    riv_m = margins(rival_clip_embs, anchor, rival_centroids, cohort)
    pos_lo = float(np.percentile(pos_m, 10))
    riv_hi = float(riv_m.max())
    delta = max(DELTA_FLOOR, (pos_lo + riv_hi) / 2.0)
    if riv_hi >= pos_lo:
        log(f"  WARNING: rival margins overlap positives "
            f"(riv max {riv_hi:+.3f} vs pos p10 {pos_lo:+.3f}) - "
            f"a lexicon neighbour is very close; delta floored at "
            f"{DELTA_FLOOR:+.3f}")
        delta = DELTA_FLOOR
    log(f"  margin calibration: positives p10 {pos_lo:+.3f}, "
        f"rival clips max {riv_hi:+.3f} -> delta {delta:+.3f}")
    return delta, {"pos_margin_p10": pos_lo, "pos_margin_median":
                   float(np.median(pos_m)), "rival_margin_max": riv_hi,
                   "rival_margin_median": float(np.median(riv_m))}


def margin_alarm(window_embs, anchor, rival_centroids, delta, cohort=None):
    """Contamination check for calibration audio: True if any supposedly
    keyword-free window wins the rival contest like a real detection."""
    return bool((margins(window_embs, anchor, rival_centroids, cohort) >= max(
        delta, 0.05)).any())


# --------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------

def rivals_path(keyword, project_root):
    return os.path.join(project_root, "keywords",
                        f"{keyword}_rivals{artifact_suffix()}.npz")


def save_rivals(path, keyword, words, centroids, delta, stats):
    np.savez(path, keyword=keyword, words=np.array(words),
             centroids=centroids.astype(np.float32),
             delta=np.float32(delta),
             **{k: np.float32(v) for k, v in stats.items()})


def load_rivals(path):
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=False)
    return (list(z["words"]), z["centroids"], float(z["delta"]))


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------

def cluster_events(detections, gap=CLUSTER_GAP_S):
    """Group threshold-crossing windows into detection events (by time)."""
    events, cur = [], []
    for d in sorted(detections, key=lambda d: d["time"]):
        if cur and d["time"] - cur[-1]["time"] > gap:
            events.append(cur)
            cur = []
        cur.append(d)
    if cur:
        events.append(cur)
    return events


VERIFY_SCALES = (0.8, 1.0)                     # full-word verification scales
VERIFY_OFFSETS = (-0.25, -0.15, -0.05, 0.0, 0.05, 0.15, 0.25)   # seconds


def _event_window_embeddings(y, event, window_samples, model,
                             step_seconds=0.05):
    """Embed FULL-WORD-scale windows around the event's best moment through
    the detector's own chunk-context pooling path.

    Detection windows exist to find the moment and are often sub-word
    (0.6-scale); verifying on them replays the partial-window problem (a
    0.6-scale window on "administration" is acoustically "-istration" and
    loses to '-ation' rivals). Verification instead searches a small grid of
    anchor-native-scale windows around the trigger: a true keyword wins the
    rival contest at some full-word alignment; a confusable wins at none -
    so the event margin is the MAX over this grid.
    """
    from embedders import FRAME_STRIDE, samples_to_frames
    chunk_frames = model.frames(y.astype(np.float32))
    hop_frames = max(1, int(round(step_seconds * SAMPLE_RATE / FRAME_STRIDE)))
    best = max(event, key=lambda d: d["score"])
    embs = []
    for scale in VERIFY_SCALES:
        ws = max(int(window_samples * scale), int(0.15 * SAMPLE_RATE))
        wf = samples_to_frames(ws)
        pooled, start_frames = model.pool_windows(chunk_frames, wf, hop_frames)
        starts_s = start_frames * FRAME_STRIDE / SAMPLE_RATE
        for off in VERIFY_OFFSETS:
            idx = int(np.argmin(np.abs(starts_s - (best["time"] + off))))
            embs.append(pooled[idx])
    return l2_normalize(np.stack(embs), axis=1)


def verify_chunk(y, detections, window_samples, model, anchor,
                 rival_words, rival_centroids, delta, cohort=None):
    """Any event whose median top-window margin clears delta keeps the chunk.

    Returns (ok, best_margin, nearest_rival_of_best_event).
    """
    best_margin, best_rival, ok = -np.inf, None, False
    for event in cluster_events(detections):
        embs = _event_window_embeddings(y, event, window_samples, model)
        m = margins(embs, anchor, rival_centroids, cohort)
        event_margin = float(m.max())      # best full-word alignment decides
        if event_margin > best_margin:
            best_margin = event_margin
            r_idx = int((embs @ rival_centroids.T).max(axis=0).argmax())
            best_rival = rival_words[r_idx]
        if event_margin >= delta:
            ok = True
    return ok, best_margin, best_rival
