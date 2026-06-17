"""Per-keyword detection threshold for the PhonMatchNet detector.

Scores the keyword against a cohort of NEGATIVE audio (speech that does NOT
contain the keyword) and sets the threshold at a high percentile of those
negative per-chunk scores — i.e. a target false-alarm rate. Writes the value
to calibration.json keyed by keyword; detect.py then uses it automatically.

    python calibrate.py --keyword trump --cohort-dir <negative_audio_dir>

The cohort should be generic speech that does NOT contain the keyword (e.g. a
held-out pool of stream chunks). The more cohort chunks, the more stable the
percentile. Runs in the .venv-g2p (same env as detect.py).

NOTE: this fixes the per-keyword *threshold scale* (penalty~0.99 vs trump~0.2).
It does NOT fix ranking inversions (a confusable out-scoring the keyword, e.g.
"public" vs "published") — those need a stronger model, not a threshold.
"""
import argparse
import glob
import json
import os

import librosa
import numpy as np

from detect import CALIB, SR, compute_gembs, load_model, score_chunks


def main():
    ap = argparse.ArgumentParser(description="Calibrate a per-keyword threshold.")
    ap.add_argument("--keyword", required=True)
    ap.add_argument("--cohort-dir", required=True,
                    help="folder of audio that does NOT contain the keyword")
    ap.add_argument("--fa-percentile", type=float, default=99.0,
                    help="threshold = this percentile of negative chunk scores")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.cohort_dir, "*.wav")))
    if not files:
        ap.error(f"no .wav files in {args.cohort_dir}")
    chunks = {os.path.basename(f): librosa.load(f, sr=SR)[0] for f in files}
    print(f"Cohort: {len(chunks)} negative chunks from {args.cohort_dir}")

    gemb_data = compute_gembs(chunks)
    model = load_model()
    scores = np.array(list(score_chunks(model, gemb_data, args.keyword).values()))
    thr = float(np.percentile(scores, args.fa_percentile))
    print(f"negative chunk scores: min={scores.min():.3f} max={scores.max():.3f} "
          f"mean={scores.mean():.3f}  ->  p{args.fa_percentile:g} threshold = {thr:.3f}")

    calib = {}
    if os.path.exists(CALIB):
        with open(CALIB) as f:
            calib = json.load(f)
    calib[args.keyword.lower()] = round(thr, 4)
    with open(CALIB, "w") as f:
        json.dump(calib, f, indent=2, sort_keys=True)
    print(f"saved threshold {thr:.3f} for '{args.keyword}' -> {CALIB}")


if __name__ == "__main__":
    main()
