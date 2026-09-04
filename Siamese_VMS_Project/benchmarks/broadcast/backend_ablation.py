"""Backend ablation: mean-pool WavLM against the trained attentive head.

The paper's Section 5.5 compares the frozen WavLM layer-10 mean-pool backend
with the trained attentive head on the same five Set D keywords, same
calibration protocol, detector alone. The trained-head arm was rebuilt under
the seeded synthesis path on 2026-09-04 and reproduces. The mean-pool arm did
not exist in reproducible form: its detections in
`logs/backup_setD_wavlm10/` are dated 2026-07-03, from before the SpeechT5
seeding fix, and the anchors behind them were discarded with the other
pre-fix builds. Recomputing micro-F1 from those stored files gives 1/0/8,
F1 0.20, matching the ledger -- but nothing on disk can regenerate them.

This rebuilds that arm from text with seeding in force, so both sides of the
ablation meet the same standard as the headline broadcast result. If the
mean-pool F1 moves away from 0.20, that is a finding, not something to
repair.

WHAT IS HELD FIXED, and why each one matters:
  * the five keywords and Chunkset_D, matching the ledger's row;
  * the enrollment recipe -- pipeline defaults, 7 canonical voices, no extra
    arguments, exactly as `harness.py` establishes for the broadcast track;
  * the calibration protocol -- defaults, fa-percentile 100, safety-margin
    0.2, top-k 50;
  * the ground-truth rule, recomputed by the pipeline's own stem-family
    matcher from each chunkset's transcripts.

WHAT CHANGES: `SIAMESE_BACKEND` only. It must be set before anything imports
`core/scoring.py`, which resolves the backend, the artifact suffix and the
project root at import time; `common/runtime.py` bakes `wavlm-trained` into
the image environment, so the override happens here and is passed down to
every stage subprocess.

RIVAL STAGES ARE SKIPPED. The comparison in the ledger is detector-alone, and
the rival bank exists only for the `_wavlm10ft` suffix -- `rival_builder`
under the mean-pool backend would find no bank and silently degrade to
lexicon-only rivals, measuring something nobody asked about.

Artifacts land under a per-backend root and never touch the run used by the
headline results.

Run:
    modal run backend_ablation.py::run_arm --backend wavlm
    modal run backend_ablation.py::score_arm --backend wavlm
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modal

from common.runtime import REPO, data, image, runtime  # noqa: F401

app = modal.App("vms-backend-ablation")
app.include(runtime)

BROADCAST = "/data/broadcast"
ABLATION_ROOT = "/data/run_ablation"

# The ledger's row: Set D, five keywords, detector alone.
EVAL = [("Chunkset_D", "russia"), ("Chunkset_D", "weather"),
        ("Chunkset_D", "scotland"), ("Chunkset_D", "ireland"),
        ("Chunkset_D", "brighten")]

# Detector path only - no rival_builder, no verify_detections.
STAGES = [
    ("keyword_generator", "keywords/{kw}_anchor{suf}.npz"),
    ("cohort_builder", "keywords/cohort_{kw}{suf}.npz"),
    ("calibrate", "keywords/{kw}_calibration{suf}.json"),
    ("detector", "logs/detections_{kw}.json"),
]


@app.function(image=image, volumes={"/data": data}, cpu=4.0, memory=16384,
              timeout=7200, retries=1, max_containers=8)
def build(job: dict):
    """One keyword end to end under one backend."""
    import subprocess
    import time

    kw = str(job["keyword"]).lower().strip()
    chunkset = job["chunkset"]
    backend = job["backend"]
    root = job["root"]
    audio_dir = f"{BROADCAST}/{chunkset}"

    # Order matters: scoring.py reads all three at import time.
    os.environ["SIAMESE_BACKEND"] = backend
    os.environ["SIAMESE_PROJECT_ROOT"] = root
    os.environ["SIAMESE_AUDIO_DIR"] = audio_dir
    env = dict(os.environ)
    os.makedirs(f"{root}/keywords", exist_ok=True)
    os.makedirs(f"{root}/logs", exist_ok=True)

    sys.path.insert(0, "/app/core")
    from scoring import BACKEND, artifact_suffix           # noqa: E402
    assert BACKEND == backend, (
        f"backend override did not take: scoring sees {BACKEND!r}, "
        f"expected {backend!r}. The image bakes SIAMESE_BACKEND, so this "
        f"must be set before the first import of scoring.")
    suf = artifact_suffix()

    if not os.path.exists(f"{audio_dir}/transcripts.txt"):
        return {"keyword": kw, "ok": False, "stage": "preflight",
                "err": f"no transcripts.txt in {audio_dir}"}

    timings = {}
    for name, tpl in STAGES:
        artifact = os.path.join(root, tpl.format(kw=kw, suf=suf))
        t = time.perf_counter()
        p = subprocess.run(
            [sys.executable, f"/app/pipeline/{name}.py", "--keyword", kw],
            cwd="/app", env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace")
        timings[name] = round(time.perf_counter() - t, 1)
        if p.returncode != 0 or not os.path.exists(artifact):
            data.commit()
            return {"keyword": kw, "ok": False, "stage": name,
                    "timings": timings, "suffix": suf,
                    "err": (p.stderr or p.stdout or "")[-1500:]}

    with open(os.path.join(root, "keywords",
                           f"{kw}_calibration{suf}.json")) as f:
        cal = json.load(f)
    data.commit()
    return {"keyword": kw, "ok": True, "backend": backend, "suffix": suf,
            "timings": timings, "threshold": cal.get("threshold")}


@app.function(image=image, volumes={"/data": data}, timeout=1800)
def score(root: str, backend: str):
    """Micro and macro P/R/F1 over the five keywords, project ground truth."""
    import re

    os.environ["SIAMESE_BACKEND"] = backend
    os.environ["SIAMESE_PROJECT_ROOT"] = root
    sys.path.insert(0, "/app/core")
    from scoring import keyword_in_tokens, spoken_numbers  # noqa: E402

    def truth(chunkset, keyword):
        out, cur = {}, None
        with open(f"{BROADCAST}/{chunkset}/transcripts.txt",
                  encoding="utf-8") as f:
            for line in f:
                m = re.match(r"\[(live_\d+\.wav)\]", line.strip())
                if m:
                    cur = m.group(1)
                    out.setdefault(cur, False)
                elif cur:
                    toks = re.findall(r"[a-z']+", spoken_numbers(line.lower()))
                    if keyword_in_tokens(toks, keyword):
                        out[cur] = True
        return out

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)

    rows, TP, FP, FN = [], 0, 0, 0
    for chunkset, kw in EVAL:
        path = os.path.join(root, "logs", f"detections_{kw}.json")
        if not os.path.exists(path):
            rows.append({"keyword": kw, "missing": True})
            continue
        with open(path) as f:
            det = json.load(f)
        t = truth(chunkset, kw)
        tp = fp = fn = 0
        for c in det["chunks"]:
            fired, real = bool(c.get("detections")), t.get(c["file"], False)
            tp += fired and real
            fp += fired and not real
            fn += real and not fired
        TP, FP, FN = TP + tp, FP + fp, FN + fn
        p, r, f1 = prf(tp, fp, fn)
        rows.append({"keyword": kw, "tp": tp, "fp": fp, "fn": fn,
                     "p": round(p, 3), "r": round(r, 3), "f1": round(f1, 3),
                     "threshold": det.get("threshold"),
                     "n_chunks": len(det["chunks"])})
    p, r, f1 = prf(TP, FP, FN)
    scored = [x for x in rows if not x.get("missing")]
    out = {"backend": backend, "root": root, "rows": rows,
           "micro": {"tp": TP, "fp": FP, "fn": FN, "p": round(p, 3),
                     "r": round(r, 3), "f1": round(f1, 3)},
           "macro_f1": round(sum(x["f1"] for x in scored) / len(scored), 3)
           if scored else None,
           "n_decisions": sum(x.get("n_chunks", 0) for x in scored)}

    os.makedirs("/data/ablation", exist_ok=True)
    with open(f"/data/ablation/backend_{backend}.json", "w") as f:
        json.dump(out, f, indent=2)
    data.commit()
    print(json.dumps(out, indent=2))
    return out


# The calibration protocol axes. "aligned" embeds negatives through the
# detector's chunk-pooled path; "guard" excludes keyword-bearing chunks. The
# reported protocol is aligned+guard; the other three reproduce the two bugs
# fixed on 2026-07-03, which until now were quoted from the ledger with no
# artifact behind them. Isolated cells cap the negative count because they
# embed one window per forward pass; aligned cells keep the pipeline default,
# which is exhaustive over available windows anyway (2,316-6,320 per keyword).
def _suffix(backend, backbone="microsoft/wavlm-base-plus", layer=10):
    """Mirror of core/scoring.py::artifact_suffix, computed from arguments.

    Kept in step with that function by hand. It exists because scoring.py
    binds the backend at import time and Modal containers are reused.
    """
    if backend == "baseline":
        return ""
    short = backbone.split("/")[-1].replace("-base-plus", "").replace("-", "")
    suffix = f"_{short}{layer}"
    if backend == "wavlm-trained":
        suffix += "ft"
    return suffix


VARIANTS = {
    "aligned_guard":     {"flags": [], "negatives": 40000},
    "isolated_guard":    {"flags": ["--isolated-negatives"], "negatives": 4000},
    "aligned_noguard":   {"flags": ["--no-guard"], "negatives": 40000},
    "isolated_noguard":  {"flags": ["--isolated-negatives", "--no-guard"],
                          "negatives": 4000},
}


@app.function(image=image, volumes={"/data": data}, cpu=4.0, memory=16384,
              timeout=7200, retries=1, max_containers=10)
def variant(job: dict):
    """Re-calibrate and re-detect one keyword under one calibration protocol.

    The anchor and cohort are copied from the backend's base root and never
    rebuilt, so the only thing varying across cells is the calibration.
    """
    import re
    import shutil
    import subprocess
    import time

    kw = str(job["keyword"]).lower().strip()
    chunkset, backend = job["chunkset"], job["backend"]
    vname, src_root, root = job["variant"], job["src_root"], job["root"]
    spec = VARIANTS[vname]
    audio_dir = f"{BROADCAST}/{chunkset}"

    os.environ["SIAMESE_BACKEND"] = backend
    os.environ["SIAMESE_PROJECT_ROOT"] = root
    os.environ["SIAMESE_AUDIO_DIR"] = audio_dir
    env = dict(os.environ)
    os.makedirs(f"{root}/keywords", exist_ok=True)
    os.makedirs(f"{root}/logs", exist_ok=True)

    sys.path.insert(0, "/app/core")
    from scoring import keyword_in_tokens, spoken_numbers    # noqa: E402
    # NOT scoring.artifact_suffix(): scoring resolves the backend at IMPORT
    # time, and Modal reuses a container across jobs, so a container that
    # already served a `wavlm` cell reports `_wavlm10` for a `wavlm-trained`
    # one and the prerequisite copy silently looks in the wrong place. This
    # mirrors artifact_suffix() from the backend string instead, which is
    # what the fresh stage subprocesses will independently compute.
    suf = _suffix(backend)

    for tpl in (f"{kw}_anchor{suf}.npz", f"cohort_{kw}{suf}.npz"):
        src = os.path.join(src_root, "keywords", tpl)
        if not os.path.exists(src):
            return {"keyword": kw, "variant": vname, "ok": False,
                    "err": f"missing prerequisite {src}"}
        shutil.copyfile(src, os.path.join(root, "keywords", tpl))

    timings = {}
    for name, extra, artifact in [
        ("calibrate", ["--negatives", str(spec["negatives"])] + spec["flags"],
         f"keywords/{kw}_calibration{suf}.json"),
        ("detector", [], f"logs/detections_{kw}.json"),
    ]:
        t = time.perf_counter()
        p = subprocess.run(
            [sys.executable, f"/app/pipeline/{name}.py", "--keyword", kw] + extra,
            cwd="/app", env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace")
        timings[name] = round(time.perf_counter() - t, 1)
        if p.returncode != 0 or not os.path.exists(os.path.join(root, artifact)):
            data.commit()
            return {"keyword": kw, "variant": vname, "ok": False, "stage": name,
                    "err": (p.stderr or p.stdout or "")[-1200:]}

    with open(os.path.join(root, "keywords",
                           f"{kw}_calibration{suf}.json")) as f:
        cal = json.load(f)
    with open(os.path.join(root, "logs", f"detections_{kw}.json")) as f:
        det = json.load(f)

    cur, t = None, {}
    with open(f"{audio_dir}/transcripts.txt", encoding="utf-8") as f:
        for line in f:
            m = re.match(r"\[(live_\d+\.wav)\]", line.strip())
            if m:
                cur = m.group(1)
                t.setdefault(cur, False)
            elif cur:
                toks = re.findall(r"[a-z']+", spoken_numbers(line.lower()))
                if keyword_in_tokens(toks, kw):
                    t[cur] = True
    tp = fp = fn = 0
    for c in det["chunks"]:
        fired, real = bool(c.get("detections")), t.get(c["file"], False)
        tp += fired and real
        fp += fired and not real
        fn += real and not fired

    data.commit()
    return {"keyword": kw, "variant": vname, "backend": backend, "ok": True,
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "threshold": cal.get("threshold"), "n_neg": cal.get("n_neg"),
            "guard": cal.get("guard"), "aligned": cal.get("negatives_aligned"),
            "n_chunks": len(det["chunks"]), "timings": timings}


@app.local_entrypoint()
def sweep(backends: str = "wavlm,wavlm-trained"):
    """The backend x calibration-protocol grid."""
    def prf(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0)

    jobs, out = [], {}
    for backend in [b.strip() for b in backends.split(",") if b.strip()]:
        src = f"{ABLATION_ROOT}_{backend.replace('-', '')}"
        for vname in VARIANTS:
            for cs, kw in EVAL:
                jobs.append({"keyword": kw, "chunkset": cs, "backend": backend,
                             "variant": vname, "src_root": src,
                             "root": f"{src}_{vname}"})
    print(f"{len(jobs)} cells")
    rows = list(variant.map(jobs, order_outputs=False))
    for r in rows:
        if not r.get("ok"):
            print(f"  FAIL {r.get('backend')} {r.get('variant')} "
                  f"{r.get('keyword')}: {str(r.get('err'))[:200]}")
    grid = {}
    for r in rows:
        if not r.get("ok"):
            continue
        key = (r["backend"], r["variant"])
        g = grid.setdefault(key, {"tp": 0, "fp": 0, "fn": 0, "f1s": [],
                                  "n_neg": [], "thr": {}})
        g["tp"] += r["tp"]; g["fp"] += r["fp"]; g["fn"] += r["fn"]
        g["f1s"].append(prf(r["tp"], r["fp"], r["fn"])[2])
        g["n_neg"].append(r["n_neg"])
        g["thr"][r["keyword"]] = r["threshold"]
    print(f"\n{'backend':14s}{'protocol':20s}{'TP/FP/FN':>10s}{'micro F1':>10s}"
          f"{'macro F1':>10s}{'neg/kw':>12s}")
    print("-" * 76)
    for (b, v), g in sorted(grid.items()):
        p, r, f1 = prf(g["tp"], g["fp"], g["fn"])
        nn = g["n_neg"]
        counts = f"{g['tp']}/{g['fp']}/{g['fn']}"
        macro = sum(g["f1s"]) / len(g["f1s"])
        negs = f"{min(nn)}-{max(nn)}"
        print(f"{b:14s}{v:20s}{counts:>10s}{f1:>10.3f}{macro:>10.3f}"
              f"{negs:>12s}")
        out[f"{b}|{v}"] = {"backend": b, "protocol": v, "tp": g["tp"],
                           "fp": g["fp"], "fn": g["fn"], "micro_p": round(p, 3),
                           "micro_r": round(r, 3), "micro_f1": round(f1, 3),
                           "macro_f1": round(sum(g["f1s"]) / len(g["f1s"]), 3),
                           "n_neg_min": min(nn), "n_neg_max": max(nn),
                           "thresholds": g["thr"]}
    write_grid.remote(out)


@app.function(image=image, volumes={"/data": data}, timeout=600)
def write_grid(out: dict):
    os.makedirs("/data/ablation", exist_ok=True)
    with open("/data/ablation/backend_x_calibration.json", "w") as f:
        json.dump({"keywords": [kw for _, kw in EVAL], "n_decisions": 50,
                   "cells": out}, f, indent=2)
    data.commit()
    print("wrote /data/ablation/backend_x_calibration.json")


@app.local_entrypoint()
def run_arm(backend: str = "wavlm"):
    root = f"{ABLATION_ROOT}_{backend.replace('-', '')}"
    jobs = [{"keyword": kw, "chunkset": cs, "backend": backend, "root": root}
            for cs, kw in EVAL]
    print(f"backend {backend} -> {root}, {len(jobs)} keywords")
    ok = 0
    for res in build.map(jobs, order_outputs=False):
        flag = "ok " if res.get("ok") else "FAIL"
        ok += bool(res.get("ok"))
        print(f"  [{flag}] {res['keyword']:10s} {res.get('timings', {})}"
              f"{'' if res.get('ok') else '  ' + str(res.get('err'))[:300]}")
    print(f"{ok}/{len(jobs)} built")


@app.local_entrypoint()
def score_arm(backend: str = "wavlm"):
    root = f"{ABLATION_ROOT}_{backend.replace('-', '')}"
    score.remote(root, backend)
