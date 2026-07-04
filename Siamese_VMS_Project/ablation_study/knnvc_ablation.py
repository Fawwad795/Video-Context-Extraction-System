"""Does kNN-VC domain conversion still matter with wavlm-trained + combined_best?

Motivated by a bug found after the pipeline-parameter ablation: convert_anchor_knnvc.py's
--holdout default (6) was out of sync with keyword_generator.py's new default (4), so with
only 7 canonical voices the kNN-VC conversion silently no-op'd (bare `return`, exit 0) for
the voices_7/combined_best configs - meaning those F1=1.00 results were actually scored on
the raw TTS anchor, never on the domain-converted one.

Now that convert_anchor_knnvc.py's default is fixed and its abort paths exit non-zero, this
script runs BOTH conditions for real on 3 fresh keywords (not in the original 5-keyword
ablation set, not previously tested): sunny (live_0), windy (live_3), showers (live_6).

For each keyword, in one isolated SIAMESE_PROJECT_ROOT (cache-free, same protocol as
run_ablation.py):
  1. keyword_generator.py -> TTS anchor
  2. cohort_builder + calibrate + detector on the TTS anchor      -> "no_knnvc" metrics
  3. convert_anchor_knnvc.py -> overwrites the anchor in place
  4. cohort_builder + calibrate + detector on the converted anchor -> "with_knnvc" metrics

Usage (from Siamese_VMS_Project):
  python ablation_study/knnvc_ablation.py
"""

import json
import os
import sys

ABLATION_DIR = os.path.dirname(os.path.abspath(__file__))
if ABLATION_DIR not in sys.path:
    sys.path.insert(0, ABLATION_DIR)

from run_ablation import env_for_run, eval_keyword, prepare_run_root, run_step

KEYWORDS = ["sunny", "windy", "showers"]

PARAMS = {
    "n_random": 0, "n_blend": 0, "n_augment": 2, "holdout": 4, "kg_seed": 42,
    "stream_windows": 50, "tts_words": 0, "no_tts": True,
    "cohort_seed": 123, "negatives": 40000, "fa_percentile": 100, "calib_seed": 777,
}

RESULTS_PATH = os.path.join(ABLATION_DIR, "results", "knnvc_ablation.jsonl")


def run_cohort_calib_detect(keyword, env, p):
    steps = {}
    args_cb = [
        "--keyword", keyword,
        "--stream-windows", str(p["stream_windows"]),
        "--tts-words", str(p["tts_words"]),
        "--seed", str(p["cohort_seed"]),
    ]
    if p.get("no_tts"):
        args_cb.append("--no-tts")
    steps["cohort_builder"] = run_step("cohort_builder.py", args_cb, env)

    args_cal = [
        "--keyword", keyword,
        "--negatives", str(p["negatives"]),
        "--fa-percentile", str(p["fa_percentile"]),
        "--seed", str(p["calib_seed"]),
    ]
    steps["calibrate"] = run_step("calibrate.py", args_cal, env)
    steps["detector"] = run_step("detector.py", ["--keyword", keyword], env)
    return steps


def main():
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    records = []

    for kw in KEYWORDS:
        print(f"\n=== {kw} ===")
        run_root = prepare_run_root(f"knnvc_{kw}")
        env = env_for_run(run_root)

        t_kg = run_step("keyword_generator.py", [
            "--keyword", kw,
            "--n-random", str(PARAMS["n_random"]),
            "--n-blend", str(PARAMS["n_blend"]),
            "--n-augment", str(PARAMS["n_augment"]),
            "--holdout", str(PARAMS["holdout"]),
            "--seed", str(PARAMS["kg_seed"]),
        ], env)

        # Condition A: raw TTS anchor, no kNN-VC
        steps_no = run_cohort_calib_detect(kw, env, PARAMS)
        metrics_no = eval_keyword(kw, run_root)
        time_no = t_kg + sum(steps_no.values())
        print(f"  no_knnvc:   F1={metrics_no['f1']:.2f}  "
              f"P={metrics_no['p']:.2f} R={metrics_no['r']:.2f}  "
              f"TP={metrics_no['tp']} FP={metrics_no['fp']} FN={metrics_no['fn']}  "
              f"time={time_no:.0f}s")

        # Convert the anchor into the stream voice
        t_knnvc = run_step("convert_anchor_knnvc.py", [
            "--keyword", kw, "--holdout", str(PARAMS["holdout"]),
        ], env)

        # Condition B: kNN-VC converted anchor
        steps_yes = run_cohort_calib_detect(kw, env, PARAMS)
        metrics_yes = eval_keyword(kw, run_root)
        time_yes = t_kg + t_knnvc + sum(steps_yes.values())
        print(f"  with_knnvc: F1={metrics_yes['f1']:.2f}  "
              f"P={metrics_yes['p']:.2f} R={metrics_yes['r']:.2f}  "
              f"TP={metrics_yes['tp']} FP={metrics_yes['fp']} FN={metrics_yes['fn']}  "
              f"time={time_yes:.0f}s")

        record = {
            "keyword": kw,
            "no_knnvc": {**metrics_no, "seconds": round(time_no, 1)},
            "with_knnvc": {**metrics_yes, "seconds": round(time_yes, 1)},
        }
        records.append(record)
        with open(RESULTS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def micro(cond):
        tp = sum(r[cond]["tp"] for r in records)
        fp = sum(r[cond]["fp"] for r in records)
        fn = sum(r[cond]["fn"] for r in records)
        p = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * rec / (p + rec) if p + rec else 0.0
        return p, rec, f1

    print(f"\n{'keyword':10s} {'no_knnvc F1':12s} {'with_knnvc F1':13s}")
    for r in records:
        print(f"{r['keyword']:10s} {r['no_knnvc']['f1']:<12.2f} {r['with_knnvc']['f1']:<13.2f}")
    p_no, r_no, f1_no = micro("no_knnvc")
    p_yes, r_yes, f1_yes = micro("with_knnvc")
    print(f"\nmicro (n={len(records)} keywords x 10 chunks):")
    print(f"  no_knnvc:   P={p_no:.2f} R={r_no:.2f} F1={f1_no:.2f}")
    print(f"  with_knnvc: P={p_yes:.2f} R={r_yes:.2f} F1={f1_yes:.2f}")


if __name__ == "__main__":
    main()
