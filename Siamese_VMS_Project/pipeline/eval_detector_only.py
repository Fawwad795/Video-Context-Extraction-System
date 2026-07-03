"""Detector-only chunk metrics: did stage 2 cross the threshold by itself?

Step 3's success criterion is that REAL keyword windows clear the
calibrated AS-norm threshold directly, instead of arriving below it and
being rescued by the phoneme verifier. This script scores the detector's
raw threshold decisions (logs/detections_<kw>.json, `detections` field -
NOT the verifier output) against the transcript ground truth, per keyword
and micro/macro across keywords.

Ground truth: keyword appears as a token in audios/transcripts.txt for
the chunk (same rule as eval_scoring_ab.py / validate_detection.py, but
without re-running Whisper).

Run under the backend being evaluated, e.g.:
    SIAMESE_BACKEND=wavlm-trained python pipeline/eval_detector_only.py \
        --keywords russia,weather,scotland,ireland,brighten
"""

import argparse
import json
import os
import re

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "core"))

from scoring import PROJECT_ROOT, list_chunk_audios


def load_truth(keyword):
    path = os.path.join(PROJECT_ROOT, "audios", "transcripts.txt")
    truth, current = {}, None
    for line in open(path, encoding="utf-8"):
        m = re.match(r"\[(live_\d+\.wav)\]", line.strip())
        if m:
            current = m.group(1)
            truth.setdefault(current, False)
        elif current:
            tokens = set(re.findall(r"[a-z']+", line.lower()))
            if keyword.lower() in tokens:
                truth[current] = True
    return truth


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--keywords", default="russia,weather,scotland,ireland")
    ap.add_argument("--suffix", default="",
                    help="detections file suffix, e.g. _unverified")
    ap.add_argument("--details", action="store_true",
                    help="print per-chunk rows")
    args = ap.parse_args()

    chunk_names = [os.path.basename(f) for f in list_chunk_audios()]
    totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    macro_f1 = []

    print(f"{'keyword':<14}{'TP':>4}{'FP':>4}{'FN':>4}{'TN':>4}"
          f"{'P':>7}{'R':>7}{'F1':>7}   margin(min TP hit - thr)")
    print("-" * 78)
    for keyword in [k.strip() for k in args.keywords.split(",")]:
        json_path = os.path.join(PROJECT_ROOT, "logs",
                                 f"detections_{keyword}{args.suffix}.json")
        if not os.path.exists(json_path):
            print(f"{keyword:<14} missing {json_path}")
            continue
        with open(json_path) as f:
            results = json.load(f)
        threshold = results["threshold"]
        truth = load_truth(keyword)
        by_file = {c["file"]: c for c in results["chunks"]}

        tp = fp = fn = tn = 0
        true_hit_scores = []
        for name in chunk_names:
            t = truth.get(name, False)
            c = by_file.get(name, {})
            pred = bool(c.get("detections"))
            if t and pred:
                tp += 1
                true_hit_scores.append(max(d["score"] for d in c["detections"]))
            elif t:
                fn += 1
            elif pred:
                fp += 1
            else:
                tn += 1
            if args.details:
                print(f"  {name:<14} truth={str(t):<6} pred={str(pred):<6} "
                      f"best={c.get('best_score', float('nan')):.2f} "
                      f"thr={threshold:.2f}")
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        macro_f1.append(f1)
        for k, v in zip(("tp", "fp", "fn", "tn"), (tp, fp, fn, tn)):
            totals[k] += v
        margin = (f"{min(true_hit_scores) - threshold:+.2f}"
                  if true_hit_scores else "  n/a")
        print(f"{keyword:<14}{tp:>4}{fp:>4}{fn:>4}{tn:>4}"
              f"{p:>7.2f}{r:>7.2f}{f1:>7.2f}   {margin}")

    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    print("-" * 78)
    print(f"{'micro':<14}{tp:>4}{fp:>4}{fn:>4}{totals['tn']:>4}"
          f"{p:>7.2f}{r:>7.2f}{f1:>7.2f}")
    if macro_f1:
        print(f"{'macro F1':<14}{'':>28}{sum(macro_f1) / len(macro_f1):>7.2f}")


if __name__ == "__main__":
    main()
