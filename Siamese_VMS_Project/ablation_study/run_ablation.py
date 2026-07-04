"""Cache-free pipeline parameter ablation for Siamese KWS.

Each config gets an isolated SIAMESE_PROJECT_ROOT under runs/<config_id>/.
Audios + transcripts.txt are copied from audios/; keywords/ and logs/ are
created fresh (no variant wav reuse).

Runs the full setup pipeline for five Set D keywords, times each step, and
records micro-averaged P/R/F1 using audios/transcripts.txt ground truth
(same rule as reports/EXPERIMENT_LOG.md).

Usage (from Siamese_VMS_Project):
  python ablation_study/run_ablation.py --configs all
  python ablation_study/run_ablation.py --configs baseline voices_7
  python ablation_study/run_ablation.py --configs combined_best
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

ABLATION_DIR = os.path.dirname(os.path.abspath(__file__))
SIAMESE_ROOT = os.path.dirname(ABLATION_DIR)
PIPELINE_DIR = os.path.join(SIAMESE_ROOT, "pipeline")
FIXTURE_AUDIO = os.path.join(SIAMESE_ROOT, "audios")
RESULTS_PATH = os.path.join(ABLATION_DIR, "results", "results.jsonl")
RUNS_DIR = os.path.join(ABLATION_DIR, "runs")
COMBINED_PATH = os.path.join(ABLATION_DIR, "results", "combined_best.json")

CHECKPOINT = os.path.join(SIAMESE_ROOT, "checkpoints", "siamese_v3_best.pth")

STEPS = (
    "keyword_generator",
    "cohort_builder",
    "calibrate",
    "detector",
)


def load_configs():
    with open(os.path.join(ABLATION_DIR, "configs.json"), encoding="utf-8") as f:
        return json.load(f)


def env_for_run(run_root):
    env = os.environ.copy()
    env["SIAMESE_PROJECT_ROOT"] = run_root
    env["SIAMESE_BACKEND"] = "wavlm-trained"
    env["SIAMESE_WEIGHTS"] = CHECKPOINT
    env["SIAMESE_V3_WEIGHTS"] = CHECKPOINT
    env.setdefault("PYTHONIOENCODING", "utf-8")
    # Models are pre-downloaded; avoid flaky hub requests during long runs.
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    return env


def prepare_run_root(config_id):
    run_root = os.path.join(RUNS_DIR, config_id)
    audio_dir = os.path.join(run_root, "audios")
    kw_dir = os.path.join(run_root, "keywords")
    log_dir = os.path.join(run_root, "logs")

    if os.path.isdir(kw_dir):
        shutil.rmtree(kw_dir)
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)

    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(kw_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    for wav in glob.glob(os.path.join(FIXTURE_AUDIO, "live_*.wav")):
        shutil.copy2(wav, audio_dir)
    transcript_src = os.path.join(FIXTURE_AUDIO, "transcripts.txt")
    if os.path.exists(transcript_src):
        shutil.copy2(transcript_src, os.path.join(audio_dir, "transcripts.txt"))

    return run_root


def run_step(script, args, env, timeout=3600):
    cmd = [sys.executable, os.path.join(PIPELINE_DIR, script)] + args
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=SIAMESE_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-4000:]
        raise RuntimeError(
            f"{script} failed (exit {proc.returncode}) after {elapsed:.1f}s:\n{tail}")
    return elapsed


def load_truth(keyword, run_root):
    path = os.path.join(run_root, "audios", "transcripts.txt")
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


def list_chunks(run_root):
    audio_dir = os.path.join(run_root, "audios")
    files = glob.glob(os.path.join(audio_dir, "live_*.wav"))
    return sorted(files, key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))


def eval_keyword(keyword, run_root):
    json_path = os.path.join(run_root, "logs", f"detections_{keyword}.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing {json_path}")
    with open(json_path, encoding="utf-8") as f:
        results = json.load(f)
    by_file = {c["file"]: bool(c.get("detections")) for c in results["chunks"]}
    truth = load_truth(keyword, run_root)

    tp = fp = fn = tn = 0
    for name in [os.path.basename(p) for p in list_chunks(run_root)]:
        t = truth.get(name, False)
        p = by_file.get(name, False)
        if t and p:
            tp += 1
        elif t:
            fn += 1
        elif p:
            fp += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "p": prec, "r": rec, "f1": f1}


def run_pipeline_for_keyword(keyword, params, env):
    step_seconds = {}
    p = params

    args_kg = [
        "--keyword", keyword,
        "--n-random", str(p["n_random"]),
        "--n-blend", str(p["n_blend"]),
        "--n-augment", str(p["n_augment"]),
        "--holdout", str(p["holdout"]),
        "--seed", str(p["kg_seed"]),
    ]
    step_seconds["keyword_generator"] = run_step("keyword_generator.py", args_kg, env)

    args_cb = [
        "--keyword", keyword,
        "--stream-windows", str(p["stream_windows"]),
        "--tts-words", str(p["tts_words"]),
        "--seed", str(p["cohort_seed"]),
    ]
    if p.get("no_tts"):
        args_cb.append("--no-tts")
    step_seconds["cohort_builder"] = run_step("cohort_builder.py", args_cb, env)

    args_cal = [
        "--keyword", keyword,
        "--negatives", str(p["negatives"]),
        "--fa-percentile", str(p["fa_percentile"]),
        "--seed", str(p["calib_seed"]),
    ]
    step_seconds["calibrate"] = run_step("calibrate.py", args_cal, env)
    step_seconds["detector"] = run_step(
        "detector.py", ["--keyword", keyword], env)

    metrics = eval_keyword(keyword, env["SIAMESE_PROJECT_ROOT"])
    return step_seconds, metrics


def aggregate_micro(per_keyword):
    tp = fp = fn = tn = 0
    for m in per_keyword.values():
        tp += m["tp"]
        fp += m["fp"]
        fn += m["fn"]
        tn += m["tn"]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "p": p, "r": r, "f1": f1}


def run_config(cfg, keywords):
    config_id = cfg["config_id"]
    print(f"\n{'=' * 60}\nConfig: {config_id}\n{'=' * 60}")
    run_root = prepare_run_root(config_id)
    env = env_for_run(run_root)

    t0 = time.perf_counter()
    per_keyword = {}
    for kw in keywords:
        print(f"  Keyword: {kw} ...")
        step_seconds, metrics = run_pipeline_for_keyword(kw, cfg["params"], env)
        per_keyword[kw] = {**metrics, "step_seconds": step_seconds}
        print(f"    F1={metrics['f1']:.2f}  "
              f"pipeline={sum(step_seconds.values()):.0f}s")

    total = time.perf_counter() - t0
    record = {
        "config_id": config_id,
        "params": cfg["params"],
        "per_keyword": per_keyword,
        "micro": aggregate_micro({k: v for k, v in per_keyword.items()}),
        "total_seconds": round(total, 2),
        "wall_clock_iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"  micro F1={record['micro']['f1']:.2f}  total={total / 60:.1f} min")
    return record


def select_configs(all_cfgs, mode, names):
    by_id = {c["config_id"]: c for c in all_cfgs}
    if mode == "all":
        return all_cfgs
    if mode == "oat":
        return [c for c in all_cfgs if c["config_id"] != "baseline"] + [
            by_id["baseline"]
        ]
    if mode == "combined_best":
        if not os.path.exists(COMBINED_PATH):
            raise FileNotFoundError(
                f"{COMBINED_PATH} not found - run aggregate_results.py --write-combined first")
        with open(COMBINED_PATH, encoding="utf-8") as f:
            combined = json.load(f)
        return [combined]
    out = []
    for n in names:
        if n not in by_id and n != "combined_best":
            raise KeyError(f"Unknown config_id: {n}")
    for n in names:
        if n == "combined_best":
            with open(COMBINED_PATH, encoding="utf-8") as f:
                out.append(json.load(f))
        else:
            out.append(by_id[n])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--configs",
        nargs="+",
        default=["all"],
        help="all | oat | combined_best | baseline voices_7 ...",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip config_ids already present in results.jsonl",
    )
    args = ap.parse_args()

    os.makedirs(os.path.join(ABLATION_DIR, "results"), exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)

    spec = load_configs()
    keywords = spec["keywords"]
    all_cfgs = spec["configs"]

    if len(args.configs) == 1 and args.configs[0] in ("all", "oat", "combined_best"):
        cfgs = select_configs(all_cfgs, args.configs[0], [])
    else:
        cfgs = select_configs(all_cfgs, "names", args.configs)

    done = set()
    if args.skip_existing and os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["config_id"])

    for cfg in cfgs:
        if cfg["config_id"] in done:
            print(f"Skipping {cfg['config_id']} (already in results.jsonl)")
            continue
        run_config(cfg, keywords)


if __name__ == "__main__":
    main()
