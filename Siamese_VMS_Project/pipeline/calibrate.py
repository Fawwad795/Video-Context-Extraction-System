"""Calibrate the per-keyword detection threshold in AS-norm units.

Instead of a hardcoded distance (the old `1.25`), the threshold is fitted
from two score distributions:

  positives: held-out TTS voices of the keyword (from keyword_generator.py),
             AS-norm scored against the centroid anchor;
  negatives: fresh random windows from the live stream chunks - real
             deployment-domain audio that does not contain the keyword.

The threshold is the empirical false-alarm operating point on the negative
score distribution (default: 100th percentile, i.e. the max negative score -
validated on Set D with 40000 negatives; see ablation_study/ for the sweep).
A percentile is used instead of mu + k*sigma because the AS-norm negative
distribution is heavily left-skewed (silence/music windows produce a long
negative tail), which makes sigma-based rules collapse.

On top of that base percentile the threshold is nudged up by a safety
margin (--safety-margin, default 0.2). The calibration negatives are a
small finite sample, so their max underestimates how high a non-keyword
window scores over hours of live audio: at detection time fresh negatives
poke above the base and sweep in as false positives, while true hits clear
it comfortably. That live tail cannot be read off the sample (a keyword
with a light sampled tail still sees live negatives well above its max), so
the margin instead uses the headroom to the positives - a fixed fraction of
the gap from the negative ceiling to the positive centre (median). It
adapts per keyword (wider separation -> wider safeguard), stays below the
positives so recall is preserved, and collapses to 0 when positives overlap
negatives. Pass --safety-margin 0 for the old pure-percentile behaviour.

Caveat: the recall estimate uses TTS positives, which are easier for a TTS
centroid anchor than real human speech - treat it as an upper bound.

Output: keywords/<keyword>_calibration.json
"""

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np

# Shared modules (scoring, siamese_model, augment_utils) live in ../core
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "core"))

import console as ui
from scoring import (DEFAULT_TOP_K, PROJECT_ROOT, SAMPLE_RATE, anchor_path,
                     asnorm_windows, calibration_path, keyword_free_chunks,
                     l2_normalize, load_cohort, load_siamese_model,
                     sample_stream_window_embeddings)


def main():
    ap = argparse.ArgumentParser(description="Calibrate the AS-norm detection threshold.")
    ap.add_argument("--keyword", default=None, help="defaults to selected_keyword.txt")
    ap.add_argument("--negatives", type=int, default=40000)
    ap.add_argument("--fa-percentile", type=float, default=100,
                    help="base threshold = this percentile of the negative scores")
    ap.add_argument("--safety-margin", type=float, default=0.2,
                    help="push the threshold up by this fraction of the gap from "
                         "the negative ceiling to the positive centre (median), as "
                         "a false-alarm safeguard; bounded below the positives so "
                         "recall is preserved. 0 disables (pure percentile).")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()

    keyword = args.keyword
    if keyword is None:
        kw_file = os.path.join(PROJECT_ROOT, "selected_keyword.txt")
        if not os.path.exists(kw_file):
            ui.fail("No --keyword given and selected_keyword.txt not found.")
            return
        keyword = open(kw_file).read().strip()

    t0 = time.perf_counter()
    ui.banner("THRESHOLD CALIBRATION", f"keyword: {keyword}")
    ui.kv("negatives", args.negatives)
    ui.kv("false-alarm percentile", f"p{args.fa_percentile:g}")

    anchor_npz = anchor_path(keyword)
    if not os.path.exists(anchor_npz):
        ui.fail(f"{anchor_npz} not found - run keyword_generator.py first.")
        return
    data = np.load(anchor_npz)
    centroid = l2_normalize(data["centroid"])
    positives = data["positives"]
    window_samples = int(data["window_samples"])

    cohort = load_cohort(keyword)
    model = load_siamese_model()

    ui.step(f"embedding {args.negatives} fresh negative stream windows ...")
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

    # Base false-alarm operating point: the requested percentile of the
    # negatives (default p100 = the max sampled negative window).
    base = float(np.percentile(neg_scores, args.fa_percentile))

    # Safety margin against live false positives. The calibration negatives
    # are a small finite sample, so `base` underestimates how high a
    # non-keyword window scores once the detector runs over hours of live
    # audio: fresh negatives poke above it and sweep in (true hits clear it
    # comfortably; false alarms enter within ~1 unit of it). That live tail
    # cannot be read off the sample - a keyword with a light sampled tail
    # (e.g. russia: max only 0.6 above p99) still saw live negatives 1.3
    # above the sampled max - so instead of extrapolating the negatives, use
    # the headroom to the positives: push the threshold a fixed fraction of
    # the way from the negative ceiling toward the positive centre. It adapts
    # per keyword (wider separation -> wider safeguard), stays below the
    # positives so recall is preserved, and collapses to 0 when positives
    # overlap negatives (a keyword inseparable from the background).
    p99 = float(np.percentile(neg_scores, 99))
    pos_median = float(np.median(pos_scores))
    gap = max(0.0, pos_median - base)
    safety = args.safety_margin * gap

    # +1e-4: the detector fires at score >= threshold and calibration windows
    # come from the detector's own scoring grid, so without the epsilon a real
    # negative window at exactly `base` would fire by float32 BLAS jitter
    # (~1e-7, shape-dependent matmul blocking); 1e-4 clears it and is
    # negligible against the AS-norm scale (units ~1).
    threshold = base + safety + 1e-4
    est_recall = float((pos_scores >= threshold).mean())

    ui.rule()
    ui.kv("cohort size / top-k", f"{cohort.shape[0]} / {args.top_k}")
    ui.kv(f"negatives (n={len(neg_scores)})",
          f"mean={neg_scores.mean():.3f} std={neg_scores.std():.3f} "
          f"p99={p99:.3f} max={neg_scores.max():.3f}")
    ui.kv(f"positives (n={len(pos_scores)})",
          f"mean={pos_scores.mean():.3f} median={pos_median:.3f} "
          f"min={pos_scores.min():.3f}")
    ui.kv(f"base p{args.fa_percentile:g}", f"{base:.3f}")
    ui.kv("safety margin",
          f"+{safety:.3f}  ({args.safety_margin:g} x gap-to-pos {gap:.3f})")
    ui.kv("threshold", f"{threshold:.3f}   (est. TTS recall {est_recall:.0%})")
    ui.item("operating points (base percentile -> TTS-positive recall):")
    for pct in (95.0, 99.0, 99.5, 100.0):
        t = float(np.percentile(neg_scores, pct))
        r = float((pos_scores >= t).mean())
        ui.item(f"  FA<={100 - pct:>4.1f}%/window  t={t:>7.3f}  recall={r:.1%}")
    headroom = float(pos_median - threshold)
    if headroom > 0:
        ui.ok(f"headroom (pos_median - threshold): {headroom:+.3f}")
    else:
        ui.warn(f"headroom (pos_median - threshold): {headroom:+.3f} - positives "
                f"overlap negatives, expect misses (domain gap)")

    out = {
        "keyword": keyword,
        "threshold": threshold,
        "base_threshold": base,
        "safety_margin": safety,
        "safety_margin_frac": args.safety_margin,
        "pos_gap": gap,
        "fa_percentile": args.fa_percentile,
        "top_k": args.top_k,
        "window_samples": window_samples,
        "neg_mean": float(neg_scores.mean()),
        "neg_std": float(neg_scores.std()),
        "neg_max": float(neg_scores.max()),
        "neg_p99": p99,
        "pos_mean": float(pos_scores.mean()),
        "pos_median": pos_median,
        "pos_min": float(pos_scores.min()),
        "pos_raw_cos_mean": float(pos_raw.mean()),
        "neg_raw_cos_mean": float(neg_raw.mean()),
        "n_pos": int(len(pos_scores)),
        "n_neg": int(len(neg_scores)),
        "est_recall": est_recall,
        "calibrated_at": datetime.now().isoformat(timespec="seconds"),
    }
    out_path = calibration_path(keyword)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    ui.done(os.path.relpath(out_path, PROJECT_ROOT), t0)
    ui.item("next: python pipeline/detector.py")


if __name__ == "__main__":
    main()
