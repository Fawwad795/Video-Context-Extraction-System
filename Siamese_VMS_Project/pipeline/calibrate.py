"""Calibrate the per-keyword detection threshold in AS-norm units.

Instead of a hardcoded distance (the old `1.25`), the threshold is fitted
from two score distributions:

  positives: held-out TTS voices of the keyword (from keyword_generator.py),
             AS-norm scored against the centroid anchor;
  negatives: fresh random windows from the live stream chunks - real
             deployment-domain audio that does not contain the keyword.

The threshold is the empirical false-alarm operating point on the negative
score distribution (default: 99.5th percentile, i.e. accept ~0.5% of
negative windows). A percentile is used instead of mu + k*sigma because the
AS-norm negative distribution is heavily left-skewed (silence/music windows
produce a long negative tail), which makes sigma-based rules collapse.

Caveat: the recall estimate uses TTS positives, which are easier for a TTS
centroid anchor than real human speech - treat it as an upper bound.

Output: keywords/<keyword>_calibration.json
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np

# Shared modules (scoring, siamese_model, augment_utils) live in ../core
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "core"))

from scoring import (DEFAULT_TOP_K, PROJECT_ROOT, SAMPLE_RATE, anchor_path,
                     asnorm_windows, calibration_path, keyword_free_chunks,
                     l2_normalize, load_cohort, load_siamese_model,
                     sample_stream_window_embeddings)


def main():
    ap = argparse.ArgumentParser(description="Calibrate the AS-norm detection threshold.")
    ap.add_argument("--keyword", default=None, help="defaults to selected_keyword.txt")
    ap.add_argument("--negatives", type=int, default=400)
    ap.add_argument("--fa-percentile", type=float, default=99.5,
                    help="threshold = this percentile of the negative scores")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()

    keyword = args.keyword
    if keyword is None:
        kw_file = os.path.join(PROJECT_ROOT, "selected_keyword.txt")
        if not os.path.exists(kw_file):
            print("No --keyword given and selected_keyword.txt not found.")
            return
        keyword = open(kw_file).read().strip()

    anchor_npz = anchor_path(keyword)
    if not os.path.exists(anchor_npz):
        print(f"{anchor_npz} not found - run keyword_generator.py first.")
        return
    data = np.load(anchor_npz)
    centroid = l2_normalize(data["centroid"])
    positives = data["positives"]
    window_samples = int(data["window_samples"])

    cohort = load_cohort(keyword)
    model = load_siamese_model()

    print(f"Embedding {args.negatives} fresh negative stream windows...")
    rng = np.random.default_rng(args.seed)
    # Leakage guard: never sample "negatives" from chunks whose transcript
    # contains the keyword - with a discriminative embedding the sampled
    # keyword windows would set the false-alarm percentile above the true
    # score (that is not a false alarm, it is the keyword). And embed them
    # via the detector's own scoring path (chunk-context frame pooling for
    # frame backends), so the fitted percentile describes the distribution
    # the detector actually thresholds.
    neg_files = keyword_free_chunks(keyword)
    neg_embs = sample_stream_window_embeddings(
        model, window_samples, args.negatives, rng, files=neg_files)

    pos_scores, pos_raw = asnorm_windows(positives, centroid, cohort, args.top_k)
    neg_scores, neg_raw = asnorm_windows(neg_embs, centroid, cohort, args.top_k)

    # Slightly above the percentile: the detector fires at score >= threshold,
    # and with calibration windows drawn from the detector's own scoring grid
    # the p100 threshold IS the score of a real negative window - which would
    # then fire by (near-)equality. The margin must exceed float32 BLAS
    # jitter between the two computations of the same window (~1e-7 observed;
    # shape-dependent matmul blocking), while staying negligible against the
    # AS-norm score scale (units ~1).
    threshold = float(np.percentile(neg_scores, args.fa_percentile)) + 1e-4
    est_recall = float((pos_scores >= threshold).mean())

    print("\n--- Calibration report ---")
    print(f"keyword: '{keyword}'  |  cohort size: {cohort.shape[0]}  |  top-k: {args.top_k}")
    print(f"negatives (n={len(neg_scores)}): mean={neg_scores.mean():.3f} "
          f"std={neg_scores.std():.3f} p99={np.percentile(neg_scores, 99):.3f} "
          f"max={neg_scores.max():.3f}")
    print(f"positives (n={len(pos_scores)}): mean={pos_scores.mean():.3f} "
          f"min={pos_scores.min():.3f}")
    print(f"threshold = p{args.fa_percentile:g}(negatives) = {threshold:.3f}")
    print("operating points (threshold -> TTS-positive recall):")
    for pct in (95.0, 99.0, 99.5, 100.0):
        t = float(np.percentile(neg_scores, pct))
        r = float((pos_scores >= t).mean())
        print(f"  FA<={100 - pct:>4.1f}%/window  t={t:>7.3f}  recall={r:.1%}")
    margin = float(pos_scores.mean() - threshold)
    print(f"margin (pos_mean - threshold): {margin:+.3f} "
          f"{'OK' if margin > 0 else '!! positives overlap negatives - expect misses (domain gap)'}")

    out = {
        "keyword": keyword,
        "threshold": threshold,
        "fa_percentile": args.fa_percentile,
        "top_k": args.top_k,
        "window_samples": window_samples,
        "neg_mean": float(neg_scores.mean()),
        "neg_std": float(neg_scores.std()),
        "neg_max": float(neg_scores.max()),
        "neg_p99": float(np.percentile(neg_scores, 99)),
        "pos_mean": float(pos_scores.mean()),
        "pos_min": float(pos_scores.min()),
        "pos_raw_cos_mean": float(pos_raw.mean()),
        "neg_raw_cos_mean": float(neg_raw.mean()),
        "n_pos": int(len(pos_scores)),
        "n_neg": int(len(neg_scores)),
        "calibrated_at": datetime.now().isoformat(timespec="seconds"),
    }
    out_path = calibration_path(keyword)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nCalibration saved: {out_path}")
    print("Next: python detector.py")


if __name__ == "__main__":
    main()
