"""Summarize ablation results and propose combined_best + sweet-spot config.

Usage (from Siamese_VMS_Project):
  python ablation_study/aggregate_results.py
  python ablation_study/aggregate_results.py --write-combined
"""

import argparse
import csv
import json
import os

ABLATION_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(ABLATION_DIR, "results", "results.jsonl")
SUMMARY_PATH = os.path.join(ABLATION_DIR, "results", "summary.csv")
COMBINED_PATH = os.path.join(ABLATION_DIR, "results", "combined_best.json")
BASELINE_PATH = os.path.join(ABLATION_DIR, "configs.json")


def load_results():
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(f"No results at {RESULTS_PATH} - run run_ablation.py first")
    rows = []
    with open(RESULTS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_baseline_params():
    with open(BASELINE_PATH, encoding="utf-8") as f:
        spec = json.load(f)
    for c in spec["configs"]:
        if c["config_id"] == "baseline":
            return dict(c["params"])
    raise KeyError("baseline config missing from configs.json")


def pareto_candidates(rows, baseline):
    b_time = baseline["total_seconds"]
    b_f1 = baseline["micro"]["f1"]
    out = []
    for r in rows:
        if r["config_id"] == "baseline":
            continue
        f1 = r["micro"]["f1"]
        dt = (b_time - r["total_seconds"]) / b_time if b_time else 0.0
        if f1 >= b_f1 - 0.02 and dt >= 0.15:
            out.append((dt, r))
    out.sort(key=lambda x: -x[0])
    return [r for _, r in out[:3]]


def merge_combined(baseline_params, candidates):
    merged = dict(baseline_params)
    for c in candidates:
        p = c["params"]
        merged["n_random"] = min(merged["n_random"], p["n_random"])
        merged["n_blend"] = min(merged["n_blend"], p["n_blend"])
        merged["n_augment"] = min(merged["n_augment"], p["n_augment"])
        merged["stream_windows"] = min(merged["stream_windows"], p["stream_windows"])
        if p.get("no_tts"):
            merged["no_tts"] = True
            merged["tts_words"] = 0
        else:
            merged["tts_words"] = min(merged["tts_words"], p["tts_words"])
        merged["negatives"] = min(merged["negatives"], p["negatives"])
    if merged["n_random"] == 0 and merged["n_blend"] == 0:
        merged["holdout"] = min(merged["holdout"], 4)
    return merged


def min_keyword_f1(row):
    return min(row["per_keyword"][k]["f1"] for k in row["per_keyword"])


def pick_sweet_spot(rows, baseline):
    b_time = baseline["total_seconds"]
    b_f1 = baseline["micro"]["f1"]
    best = None
    for r in rows:
        if r["config_id"] in ("baseline", "combined_best"):
            continue
        f1 = r["micro"]["f1"]
        dt = (b_time - r["total_seconds"]) / b_time if b_time else 0.0
        if f1 >= 0.98 or f1 >= b_f1 - 0.02:
            if dt >= 0.25 and min_keyword_f1(r) >= 0.90:
                if best is None or r["total_seconds"] < best["total_seconds"]:
                    best = r
    if best:
        return best
    combined_row = next((r for r in rows if r["config_id"] == "combined_best"), None)
    if combined_row:
        f1 = combined_row["micro"]["f1"]
        dt = (b_time - combined_row["total_seconds"]) / b_time if b_time else 0.0
        if f1 >= b_f1 - 0.02 and dt >= 0.25 and min_keyword_f1(combined_row) >= 0.90:
            return combined_row
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--write-combined",
        action="store_true",
        help="write results/combined_best.json from Pareto OAT configs",
    )
    args = ap.parse_args()

    rows = load_results()
    baseline = next((r for r in rows if r["config_id"] == "baseline"), None)
    if baseline is None:
        print("WARNING: no baseline row in results.jsonl")

    rows_sorted = sorted(rows, key=lambda r: r["total_seconds"])
    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)

    b_time = baseline["total_seconds"] if baseline else 1.0
    b_f1 = baseline["micro"]["f1"] if baseline else 0.0

    with open(SUMMARY_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "config_id", "total_min", "micro_f1", "micro_p", "micro_r",
            "delta_f1_vs_baseline", "delta_time_pct", "min_kw_f1",
        ])
        for r in rows_sorted:
            dt = (b_time - r["total_seconds"]) / b_time * 100 if baseline else 0.0
            w.writerow([
                r["config_id"],
                round(r["total_seconds"] / 60, 2),
                round(r["micro"]["f1"], 4),
                round(r["micro"]["p"], 4),
                round(r["micro"]["r"], 4),
                round(r["micro"]["f1"] - b_f1, 4) if baseline else "",
                round(dt, 1) if baseline else "",
                round(min_keyword_f1(r), 4),
            ])

    print(f"Wrote {SUMMARY_PATH}\n")
    hdr = f"{'config_id':<22}{'min':>8}{'micro_F1':>10}{'dF1':>8}{'dTime%':>9}{'min_kw':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows_sorted:
        dt = (b_time - r["total_seconds"]) / b_time * 100 if baseline else 0.0
        df1 = r["micro"]["f1"] - b_f1 if baseline else 0.0
        print(
            f"{r['config_id']:<22}"
            f"{r['total_seconds'] / 60:>8.1f}"
            f"{r['micro']['f1']:>10.2f}"
            f"{df1:>8.2f}"
            f"{dt:>9.1f}"
            f"{min_keyword_f1(r):>8.2f}"
        )

    if baseline and args.write_combined:
        candidates = pareto_candidates(rows, baseline)
        if not candidates:
            print("\nNo Pareto OAT configs; using fastest configs with F1 >= baseline - 0.02.")
            candidates = sorted(
                [r for r in rows if r["config_id"] != "baseline"
                 and r["micro"]["f1"] >= b_f1 - 0.02],
                key=lambda r: r["total_seconds"],
            )[:3]
        merged_params = merge_combined(load_baseline_params(), candidates)
        combined = {
            "config_id": "combined_best",
            "params": merged_params,
            "merged_from": [c["config_id"] for c in candidates],
        }
        with open(COMBINED_PATH, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2)
        print(f"\nWrote {COMBINED_PATH} from {combined['merged_from']}")

    if baseline:
        sweet = pick_sweet_spot(rows, baseline)
        print("\n--- Sweet-spot recommendation ---")
        if sweet:
            dt = (b_time - sweet["total_seconds"]) / b_time * 100
            print(f"Config: {sweet['config_id']}")
            print(f"  micro F1={sweet['micro']['f1']:.2f}  "
                  f"time save={dt:.1f}%  min keyword F1={min_keyword_f1(sweet):.2f}")
            print(f"  params: {json.dumps(sweet['params'], indent=2)}")
        else:
            print("No single OAT config met sweet-spot criteria (25% time save, F1 >= 0.98). "
                  "Check combined_best after running it.")


if __name__ == "__main__":
    main()
