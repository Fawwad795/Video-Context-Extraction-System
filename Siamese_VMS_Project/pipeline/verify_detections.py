"""Phoneme-level verification of detector candidates (precision stage).

The Siamese detector has high recall with the kNN-VC anchor but limited
precision: a few confusable phrases score at or above the true keyword in
embedding space. This second stage re-checks every candidate window in a
*decorrelated* view - its phoneme sequence - so a window only survives if it
actually contains the keyword's phones, not merely a similar spectral shape.

How it works:
  1. decode each candidate window with a CTC phoneme recognizer
     (facebook/wav2vec2-lv-60-espeak-cv-ft, IPA output). CTC phoneme models
     are trained against phone labels, which makes their output largely
     invariant to speaker/channel/TTS artifacts - the property the raw
     embedding space lacks;
  2. build reference phone sequences by decoding the anchor clips
     (kNN-VC-converted + clean TTS variants);
  3. score = best infix similarity  1 - lev(ref, sub(window)) / len(ref)
     over all references (window may contain neighbouring phones, so the
     reference is matched as a substring, free ends on the window side);
  4. calibrate the accept threshold like calibrate.py does: score random
     keyword-length stream windows (negatives) and take a high percentile;
  5. drop detections below the threshold and rewrite the detections JSON
     (original backed up as *_unverified.json).

Usage: python pipeline/verify_detections.py --keyword administration
Then re-run validate_detection.py to see the verified P/R/F1.
"""

import argparse
import glob
import json
import os
import shutil

import librosa
import numpy as np

# Shared modules (scoring, siamese_model, augment_utils) live in ../core
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "core"))

from scoring import PROJECT_ROOT, SAMPLE_RATE, sample_stream_windows

# IPA phones are not representable in the default Windows console codepage
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PHONEME_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"
CONTEXT_PAD_S = 0.08   # audio context added on both sides of a candidate window
MIN_REF_PHONES = 3     # discard degenerate reference decodes


def load_phoneme_model():
    # The model's Wav2Vec2PhonemeCTCTokenizer needs the phonemizer library +
    # an espeak backend just to construct, but decoding argmax ids only needs
    # the vocab table - so we load feature extractor + vocab and CTC-collapse
    # the ids ourselves.
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

    print(f"Loading phoneme recognizer: {PHONEME_MODEL} ...")
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

    Standard approximate-substring DP: deletions/insertions inside the match
    cost 1, but skipping `seq` tokens before/after the matched span is free.
    """
    n, m = len(ref), len(seq)
    if n == 0:
        return 0.0
    if m == 0:
        return 0.0
    prev = np.zeros(m + 1, dtype=np.int32)          # free start anywhere in seq
    for i in range(1, n + 1):
        cur = np.empty(m + 1, dtype=np.int32)
        cur[0] = i
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == seq[j - 1] else 1
            cur[j] = min(prev[j - 1] + cost,        # match / substitute
                         prev[j] + 1,               # delete from ref
                         cur[j - 1] + 1)            # insert from seq
        prev = cur
    dist = int(prev.min())                          # free end anywhere in seq
    return max(0.0, 1.0 - dist / n)


def best_similarity(refs, seq):
    return max((infix_similarity(r, seq) for r in refs), default=0.0)


def build_references(keyword, processor, model, torch, max_refs):
    """Decode anchor clips (kNN-VC-converted first, then clean TTS) to phones."""
    dirs = [os.path.join(PROJECT_ROOT, "keywords", f"{keyword}_variants_knnvc"),
            os.path.join(PROJECT_ROOT, "keywords", f"{keyword}_variants")]
    refs, seen = [], set()
    for d in dirs:
        for wav in sorted(glob.glob(os.path.join(d, "*.wav")))[:max_refs]:
            y, _ = librosa.load(wav, sr=SAMPLE_RATE)
            y, _ = librosa.effects.trim(y, top_db=30)
            phones = decode_phones(processor, model, torch, y.astype(np.float32))
            if len(phones) < MIN_REF_PHONES:
                continue
            key = " ".join(phones)
            tag = "knnvc" if d.endswith("_knnvc") else "tts"
            print(f"  ref [{tag}] {os.path.basename(wav)}: /{key}/")
            if key not in seen:
                seen.add(key)
                refs.append(phones)
    return refs


def main():
    ap = argparse.ArgumentParser(description="Phoneme verification of detections.")
    ap.add_argument("--keyword", default=None, help="defaults to selected_keyword.txt")
    ap.add_argument("--tau", type=float, default=None,
                    help="accept threshold; default = calibrated on stream negatives")
    ap.add_argument("--min-tau", type=float, default=0.5,
                    help="floor for the calibrated tau: a candidate must match at "
                         "least this fraction of the keyword's phones. Guards "
                         "against a lucky low negative sample (small n) letting "
                         "weak phonetic matches through.")
    ap.add_argument("--fa-percentile", type=float, default=99.0,
                    help="negative-score percentile used when calibrating tau")
    ap.add_argument("--negatives", type=int, default=40)
    ap.add_argument("--max-refs", type=int, default=8, help="reference clips per variant dir")
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()

    keyword = args.keyword
    if keyword is None:
        kw_file = os.path.join(PROJECT_ROOT, "selected_keyword.txt")
        if not os.path.exists(kw_file):
            print("No --keyword given and selected_keyword.txt not found.")
            return
        keyword = open(kw_file).read().strip()

    json_path = os.path.join(PROJECT_ROOT, "logs", f"detections_{keyword}.json")
    if not os.path.exists(json_path):
        print(f"{json_path} not found - run detector.py first.")
        return
    with open(json_path) as f:
        results = json.load(f)
    window_seconds = float(results["window_seconds"])

    processor, model, torch = load_phoneme_model()

    print(f"\nBuilding phoneme references for '{keyword}':")
    refs = build_references(keyword, processor, model, torch, args.max_refs)
    if not refs:
        print("No usable reference decodes - aborting.")
        return

    # What does a *true* match score? Leave-one-out self-similarity of the
    # references estimates it (imperfect decodes across voices/domains).
    if len(refs) > 1:
        loo = [best_similarity(refs[:i] + refs[i + 1:], r) for i, r in enumerate(refs)]
        loo_mean = float(np.mean(loo))
        print(f"reference leave-one-out self-similarity: mean={loo_mean:.3f} "
              f"min={min(loo):.3f}")
    else:
        loo_mean = 1.0

    # ---- Calibrate tau on random stream windows (same idea as calibrate.py) ----
    tau = args.tau
    if tau is None:
        rng = np.random.default_rng(args.seed)
        win = int(window_seconds * SAMPLE_RATE)
        print(f"\nCalibrating tau on {args.negatives} random stream windows...")
        neg_scores = [best_similarity(refs, decode_phones(processor, model, torch, w))
                      for w in sample_stream_windows(win, args.negatives, rng)]
        neg_scores = np.array(neg_scores)
        neg_p = float(np.percentile(neg_scores, args.fa_percentile))
        print(f"negative phone-similarity: mean={neg_scores.mean():.3f} "
              f"p90={np.percentile(neg_scores, 90):.3f} "
              f"p{args.fa_percentile:g}={neg_p:.3f} max={neg_scores.max():.3f}")
        # Midpoint between "what impostors score" and "what a true match
        # scores": with n negatives in the dozens, the raw percentile sits
        # right at the edge of the impostor distribution and near-miss
        # confusables (sharing half the phones) squeak past it.
        tau = max(0.5 * (neg_p + loo_mean), args.min_tau)
        print(f"tau = midpoint(neg_p{args.fa_percentile:g}={neg_p:.3f}, "
              f"ref_self_sim={loo_mean:.3f}), floored at {args.min_tau:.2f}")
    print(f"accept threshold tau = {tau:.3f}\n")

    # ---- Verify every candidate detection ----
    audio_dir = os.path.join(PROJECT_ROOT, "audios")
    kept_total = dropped_total = rescued_total = 0
    for chunk in results["chunks"]:
        dets = list(chunk.get("detections") or [])
        n_dets = len(dets)
        # Also verify the detector's top-N sub-threshold candidates: a true
        # keyword window whose embedding score fell below the threshold can
        # be rescued if its phone sequence matches (cascade: stage 1 recall,
        # stage 2 precision).
        for c in chunk.get("candidates") or []:
            if not any(abs(c["time"] - d["time"]) < 1e-6
                       and c.get("scale") == d.get("scale") for d in dets):
                dets.append(dict(c))
        if not dets:
            continue
        y, _ = librosa.load(os.path.join(audio_dir, chunk["file"]), sr=SAMPLE_RATE)
        verified = []
        best_sim_chunk = 0.0
        for k, det in enumerate(dets):
            ws = window_seconds * det.get("scale", 1.0)
            s = max(0, int((det["time"] - CONTEXT_PAD_S) * SAMPLE_RATE))
            e = min(len(y), int((det["time"] + ws + CONTEXT_PAD_S) * SAMPLE_RATE))
            phones = decode_phones(processor, model, torch, y[s:e].astype(np.float32))
            sim = best_similarity(refs, phones)
            det["phone_sim"] = round(sim, 3)
            det["rescued"] = k >= n_dets
            best_sim_chunk = max(best_sim_chunk, sim)
            if sim >= tau:
                verified.append(det)
        verified.sort(key=lambda d: -d["score"])
        n_rescued = sum(1 for d in verified if d["rescued"])
        rescued_total += n_rescued
        print(f"{chunk['file']}: {n_dets} detection(s) + {len(dets) - n_dets} "
              f"candidate(s) -> {len(verified)} verified"
              f"{f' ({n_rescued} rescued)' if n_rescued else ''} "
              f"(best phone-sim {best_sim_chunk:.2f})")
        kept_total += len(verified)
        dropped_total += len(dets) - len(verified)
        chunk["detections"] = verified
        chunk["phone_verified"] = True
        chunk["best_phone_sim"] = round(best_sim_chunk, 3)

    results["phone_verification"] = {
        "model": PHONEME_MODEL, "tau": tau, "n_refs": len(refs),
        "fa_percentile": args.fa_percentile,
    }

    backup = json_path.replace(".json", "_unverified.json")
    if not os.path.exists(backup):
        shutil.copy2(json_path, backup)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nkept {kept_total} detections, dropped {dropped_total}.")
    print(f"updated: {json_path}  (pre-verification copy: {backup})")
    print("Next: python pipeline/validate_detection.py --keyword", keyword)


if __name__ == "__main__":
    main()
