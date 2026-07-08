"""Phoneme-level verification of detector candidates (precision stage).

The embedding detector confuses phonetic near-neighbours of the keyword:
live monitoring for 'party' fired on "policy" (AS-norm 8.17 - above real
keyword hits at 6.13) and "public". No threshold can separate those; every
cheap structural signal (window support count, multi-scale consistency,
raw cosine) fails with them because they are all functions of the same
embedding. This module re-checks each detection in a decorrelated view -
its phone sequence - so a chunk only survives if the keyword's phones are
actually present, not merely a similar spectral shape.

Mechanism (validated on the 'party' live session, 6 TP / 3 FP):
  1. Cluster a chunk's detections into events by time (overlapping sliding
     windows fire in bursts around one utterance), then CTC-decode ONE WIDE
     span (~2.5 s) around each event with
     facebook/wav2vec2-lv-60-espeak-cv-ft (IPA output). Phone labels make
     the decode largely invariant to speaker/channel/TTS artifacts - the
     property embedding space lacks - and the wide span gives the CTC full
     acoustic context, which is what makes the decode stable: narrow
     detection-window decodes clip word edges and need alignment probing,
     and every extra probe hands a confusable another CTC-noise lottery
     ticket ("policy" reached 0.75 once across 147 narrow decodes, but its
     wide decode reads a clean /p ɔ l ə s i/).
  2. References = decoded anchor TTS variants, with a leave-one-out
     agreement filter: a TTS voice that renders the keyword badly (ksp
     spoke 'party' as /ph ɑ l i/ - which would MATCH "policy") is dropped
     as an outlier instead of poisoning the reference set.
  3. score = best infix similarity of any reference against the span's
     phones (free ends on the span side, so surrounding words cost nothing
     and morphological derivatives like 'parties' /p ɑːɹ ɾ i z/ still
     match - consistent with scoring.keyword_in_tokens' word-family rule).
  4. Chunk passes if any event's span reaches tau. On the party session
     this separates perfectly: all 6 true chunks 1.00 (the verifier even
     corrected the ground truth - whisper-base misheard live_115's
     "political party" as "policy"; whisper-small, a char-CTC and this
     phone view all agree on "party"), all 3 confusable chunks
     ("policy platform", "publicly", "public") 0.25.
  5. tau is calibrated per keyword: midpoint between what keyword-free
     stream audio scores under the SAME max-over-events statistic and the
     references' own leave-one-out self-similarity, floored at MIN_TAU.

Resurrected from the retired Stage-3 verifier (git df141b4~1) with the
outlier filter, the wide-span decode, and the platform cache added.
"""

import glob
import json
import os

import numpy as np

SAMPLE_RATE = 16000
PHONEME_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"
MIN_REF_PHONES = 3      # discard degenerate reference decodes
REF_AGREEMENT = 0.5     # drop refs whose LOO similarity to the rest is below
DECODE_SPAN_S = 2.5     # wide span decoded around each detection event
CLUSTER_GAP_S = 0.3     # detections closer than this belong to one event
MIN_TAU = 0.6           # tau floor: >= this fraction of phones must match


def load_phoneme_model():
    """Load the CTC phoneme recognizer.

    The model's Wav2Vec2PhonemeCTCTokenizer needs the phonemizer library +
    an espeak backend just to construct, but decoding argmax ids only needs
    the vocab table - so load feature extractor + vocab and CTC-collapse
    the ids manually.
    """
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

    extractor = Wav2Vec2FeatureExtractor.from_pretrained(PHONEME_MODEL)
    model = Wav2Vec2ForCTC.from_pretrained(PHONEME_MODEL)
    model.eval()
    with open(hf_hub_download(PHONEME_MODEL, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    id2tok = {i: t for t, i in vocab.items()}
    return (extractor, id2tok), model, torch


def decode_phones(processor, model, torch, audio):
    """CTC-decode a float32 16 kHz array -> list of IPA phone tokens."""
    extractor, id2tok = processor
    if len(audio) < int(0.05 * SAMPLE_RATE):
        return []
    inputs = extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    ids = torch.argmax(logits, dim=-1)[0].tolist()
    phones, prev = [], None
    for i in ids:
        if i != prev:                       # collapse CTC repeats
            tok = id2tok.get(i, "")
            if tok and not tok.startswith("<") and tok != "|":
                phones.append(tok)
        prev = i
    return phones


def infix_similarity(ref, seq):
    """1 - (min edit distance of `ref` to any substring of `seq`) / len(ref).

    Approximate-substring DP: edits inside the match cost 1, skipping seq
    tokens before/after the matched span is free.
    """
    n, m = len(ref), len(seq)
    if n == 0 or m == 0:
        return 0.0
    prev = np.zeros(m + 1, dtype=np.int32)          # free start anywhere
    for i in range(1, n + 1):
        cur = np.empty(m + 1, dtype=np.int32)
        cur[0] = i
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == seq[j - 1] else 1
            cur[j] = min(prev[j - 1] + cost,        # match / substitute
                         prev[j] + 1,               # delete from ref
                         cur[j - 1] + 1)            # insert from seq
        prev = cur
    return max(0.0, 1.0 - int(prev.min()) / n)      # free end anywhere


def best_similarity(refs, seq):
    return max((infix_similarity(r, seq) for r in refs), default=0.0)


def build_references(keyword, variants_dir, processor, model, torch,
                     max_refs=8, log=print):
    """Decode anchor TTS variants to phone references, dropping outliers.

    A TTS voice occasionally renders the keyword wrong (accented synthesis:
    ksp spoke 'party' as /ph ɑ l i/). Left in, such a reference actively
    matches the confusable the verifier exists to reject - so drop any
    reference whose best similarity to the OTHER references falls below
    REF_AGREEMENT. If everything disagrees (no consensus), keep all rather
    than none and let tau protect precision.
    """
    import librosa
    decoded = []
    for wav in sorted(glob.glob(os.path.join(variants_dir, "*.wav")))[:max_refs]:
        y, _ = librosa.load(wav, sr=SAMPLE_RATE)
        y, _ = librosa.effects.trim(y, top_db=30)
        phones = decode_phones(processor, model, torch, y.astype(np.float32))
        if len(phones) < MIN_REF_PHONES:
            continue
        decoded.append((os.path.basename(wav), phones))
    if not decoded:
        return [], 1.0

    raw = [ph for _, ph in decoded]
    loo = ([best_similarity(raw[:i] + raw[i + 1:], r) for i, r in enumerate(raw)]
           if len(raw) > 1 else [1.0])
    refs, seen = [], set()
    for (name, phones), agree in zip(decoded, loo):
        if len(raw) > 1 and agree < REF_AGREEMENT:
            log(f"  dropping outlier ref {name} /{' '.join(phones)}/ "
                f"(agreement {agree:.2f})")
            continue
        key = " ".join(phones)
        if key not in seen:
            seen.add(key)
            refs.append(phones)
            log(f"  ref {name}: /{key}/")
    if not refs:
        refs = [ph for _, ph in decoded]
    kept_loo = ([best_similarity(refs[:i] + refs[i + 1:], r)
                 for i, r in enumerate(refs)] if len(refs) > 1 else [1.0])
    return refs, float(np.mean(kept_loo))


def cluster_events(detections, gap=CLUSTER_GAP_S):
    """Group detection times into events: bursts separated by > gap seconds."""
    times = sorted(d["time"] for d in detections)
    clusters, cur = [], [times[0]]
    for t in times[1:]:
        if t - cur[-1] <= gap:
            cur.append(t)
        else:
            clusters.append(cur)
            cur = [t]
    clusters.append(cur)
    return clusters


def calibrate_tau(refs, loo_mean, window_seconds, neg_files, processor, model,
                  torch, rng, n_neg=30, decodes_per_neg=2,
                  fa_percentile=99.0, min_tau=MIN_TAU, log=print):
    """Accept threshold from impostor audio, mirroring the decision statistic.

    The chunk decision is a max over one wide-span decode per detection
    event (typically 1-2 events per chunk), so each negative draw is a max
    over the same small number of wide-span decodes of keyword-free stream
    audio.

    tau = max(midpoint(neg_percentile, loo_mean), min_tau): the percentile
    alone sits at the edge of the impostor distribution (near-miss
    confusables sharing half the phones squeak past); the references' own
    self-similarity estimates what a true match scores; the floor guards
    against a lucky low negative sample.
    """
    import librosa
    span = int(DECODE_SPAN_S * SAMPLE_RATE)
    neg_scores = []
    audios = []
    for f in neg_files:
        y, _ = librosa.load(f, sr=SAMPLE_RATE)
        if len(y) > span:
            audios.append(y)
    if not audios:
        log("  no negative audio available - tau falls back to floor")
        return float(min_tau)
    for _ in range(n_neg):
        y = audios[int(rng.integers(len(audios)))]
        best = 0.0
        for _ in range(decodes_per_neg):
            s = int(rng.integers(0, len(y) - span))
            seg = y[s:s + span].astype(np.float32)
            best = max(best, best_similarity(
                refs, decode_phones(processor, model, torch, seg)))
            if best >= 1.0:
                break
        neg_scores.append(best)
    neg_p = float(np.percentile(np.array(neg_scores), fa_percentile))
    tau = max(0.5 * (neg_p + loo_mean), float(min_tau))
    log(f"  negative phone-sim (max over {decodes_per_neg} wide decodes, "
        f"n={n_neg}): p{fa_percentile:g}={neg_p:.3f} max={max(neg_scores):.3f}")
    log(f"  tau = max(midpoint({neg_p:.3f}, {loo_mean:.3f}), {min_tau:.2f}) "
        f"= {tau:.3f}")
    return float(tau)


def verify_chunk(y, detections, window_seconds, refs, tau, processor, model,
                 torch, span=DECODE_SPAN_S, gap=CLUSTER_GAP_S):
    """Phone-verify a chunk's detections; return (passed, best_sim).

    Clusters the detections into time events, decodes ONE wide span around
    each event, and passes the chunk if any span's phone similarity reaches
    tau. The wide span gives the CTC full context (stable phones, no
    word-edge clipping, no alignment probing) and the infix matcher's free
    ends make the surrounding words cost nothing - while a confusable gets
    only one decode per event instead of one per window, so it cannot win
    a decode-noise lottery.
    """
    if not detections:
        return False, 0.0
    best = 0.0
    for cl in cluster_events(detections, gap):
        center = 0.5 * (cl[0] + cl[-1] + window_seconds)
        s = max(0, int((center - span / 2) * SAMPLE_RATE))
        e = min(len(y), int((center + span / 2) * SAMPLE_RATE))
        phones = decode_phones(processor, model, torch,
                               y[s:e].astype(np.float32))
        best = max(best, best_similarity(refs, phones))
        if best >= 1.0:
            return True, best
    return best >= tau, best


def load_phone_cache(path):
    """Load cached refs+tau; returns (refs, tau) or None."""
    if not os.path.exists(path):
        return None
    try:
        d = json.load(open(path, encoding="utf-8"))
        refs = [list(r) for r in d["refs"]]
        return (refs, float(d["tau"])) if refs else None
    except Exception:
        return None


def save_phone_cache(path, keyword, refs, tau, loo_mean):
    json.dump({"keyword": keyword, "model": PHONEME_MODEL,
               "refs": refs, "tau": tau, "loo_mean": loo_mean},
              open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
