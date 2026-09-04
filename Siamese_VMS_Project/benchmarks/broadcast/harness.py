"""Rebuild the 14 broadcast evaluation keywords under the seeded synthesis path.

Carried since 2026-08-28. The keyword builds behind the Set D/E/F results were
made before the SpeechT5 determinism fix, so their TTS clips are frozen
artifacts that no re-run can reproduce. This rebuilds them from text, with
seeding in force, and re-derives the broadcast numbers from the rebuilt
artifacts. If micro-F1 1.000 / 0.982 moves, that is a finding, not something to
repair.

Lives beside the LibriPhrase harness so it can reuse that image and volume
without a second copy of the environment: same pinned requirements, same
checkpoint, same rival bank, same offline flags.

THE RECIPE IS THE ORIGINAL ONE, NOT THE LIBRIPHRASE ONE. Established from the
artifacts on disk 2026-09-04, because the two runs differ:

  keyword_generator  defaults - 7 canonical voices, no random x-vectors.
                     `keywords/america_variants/` holds 7 files, and the
                     LibriPhrase harness's `--n-random 5` would give 12.
                     `--holdout 4` x (1 + `--n-augment 2`) = the 12 calibration
                     positives recorded in america_calibration_wavlm10ft.json.
  cohort_builder     defaults - 50 stream windows, no TTS distractors.
                     cohort_america_wavlm10ft.npz records n_stream 50, n_tts 0.
  calibrate          defaults - fa-percentile 100, safety-margin 0.2, top-k 50,
                     matching fa_percentile/safety_margin_frac/top_k in the
                     same calibration artifact.
  detector, verify   defaults.

So no stage takes extra arguments here. Passing the LibriPhrase voice budget
would rebuild a different anchor and the comparison would measure that instead
of the determinism fix.

Isolation: everything lands under /data/run_rebuild. The originals in
Siamese_VMS_Project/keywords and logs/ are never touched, on Modal or locally.

Run:
    modal volume put vms-data <repo>/audios/Chunkset_D /broadcast/Chunkset_D
    modal volume put vms-data <repo>/audios/Chunkset_E /broadcast/Chunkset_E
    modal volume put vms-data <repo>/audios/Chunkset_F /broadcast/Chunkset_F
    modal run broadcast/harness.py::rebuild_all
"""

import json
import os

import modal

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.runtime import data, image, runtime

app = modal.App("vms-broadcast")
app.include(runtime)

REBUILD_ROOT = "/data/run_rebuild"
BROADCAST = "/data/broadcast"

# Copied verbatim from Journal_Paper/experiments/compute_final_metrics.py::EVAL.
# These 14 keyword-by-set evaluations are the ones behind the paper's broadcast
# table; `island` and `party` are RAV diagnostics and are deliberately excluded.
EVAL = [("Chunkset_D", "russia"), ("Chunkset_D", "weather"),
        ("Chunkset_D", "scotland"), ("Chunkset_D", "ireland"),
        ("Chunkset_D", "brighten"), ("Chunkset_D", "outbreaks"),
        ("Chunkset_D", "insurance"), ("Chunkset_D", "western"),
        ("Chunkset_E", "administration"),
        ("Chunkset_F", "america"), ("Chunkset_F", "president"),
        ("Chunkset_F", "celebrate"), ("Chunkset_F", "fireworks"),
        ("Chunkset_F", "washington")]

# stage -> the artifact it must have written, relative to the run root.
# Exit codes are not trusted: keyword_generator returns 0 after ui.fail() when
# too few voices survive the degenerate-synthesis filter, and cohort_builder
# then "succeeds" against a default window with no anchor at all.
STAGES = [
    ("keyword_generator", "keywords/{kw}_anchor{suf}.npz"),
    ("cohort_builder", "keywords/cohort_{kw}{suf}.npz"),
    ("rival_builder", "keywords/{kw}_rivals{suf}.npz"),
    ("calibrate", "keywords/{kw}_calibration{suf}.json"),
    ("detector", "logs/detections_{kw}.json"),
    ("verify_detections", "logs/detections_{kw}_unverified.json"),
]


@app.function(image=image, volumes={"/data": data}, cpu=4.0, memory=16384,
              timeout=7200, retries=1, max_containers=16)
def rebuild(job: dict):
    """One keyword end to end: enrol, calibrate, detect, verify."""
    import subprocess
    import sys
    import time

    kw = str(job["keyword"]).lower().strip()
    chunkset = job["chunkset"]
    root = job.get("root", REBUILD_ROOT)
    audio_dir = f"{BROADCAST}/{chunkset}"

    # Set before anything imports scoring: it resolves PROJECT_ROOT and
    # AUDIO_DIR at import time.
    os.environ["SIAMESE_PROJECT_ROOT"] = root
    os.environ["SIAMESE_AUDIO_DIR"] = audio_dir
    env = dict(os.environ)
    os.makedirs(f"{root}/keywords", exist_ok=True)
    os.makedirs(f"{root}/logs", exist_ok=True)

    sys.path.insert(0, "/app/core")
    from scoring import artifact_suffix                     # noqa: E402
    suf = artifact_suffix()

    if not os.path.isdir(audio_dir):
        return {"keyword": kw, "chunkset": chunkset, "ok": False,
                "stage": "preflight", "err": f"missing audio dir {audio_dir}"}
    if not os.path.exists(f"{audio_dir}/transcripts.txt"):
        # calibrate.py's leakage guard needs it: negatives are only drawn from
        # chunks whose transcript exists and does not contain the keyword.
        return {"keyword": kw, "chunkset": chunkset, "ok": False,
                "stage": "preflight", "err": f"no transcripts.txt in {audio_dir}"}

    timings = {}
    for name, artifact_tpl in STAGES:
        artifact = os.path.join(root, artifact_tpl.format(kw=kw, suf=suf))
        t = time.perf_counter()
        p = subprocess.run(
            [sys.executable, f"/app/pipeline/{name}.py", "--keyword", kw],
            cwd="/app", env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace")
        timings[name] = round(time.perf_counter() - t, 1)
        if p.returncode != 0:
            data.commit()
            return {"keyword": kw, "chunkset": chunkset, "ok": False,
                    "stage": name, "timings": timings,
                    "err": (p.stderr or p.stdout)[-1500:]}
        if not os.path.exists(artifact):
            data.commit()
            return {"keyword": kw, "chunkset": chunkset, "ok": False,
                    "stage": name, "timings": timings,
                    "err": f"stage exited 0 but wrote no artifact: {artifact}\n"
                           f"--- stdout tail ---\n{(p.stdout or '')[-900:]}"}

    with open(os.path.join(root, "keywords",
                           f"{kw}_calibration{suf}.json")) as f:
        cal = json.load(f)
    with open(os.path.join(root, "logs", f"detections_{kw}.json")) as f:
        det = json.load(f)

    data.commit()
    return {"keyword": kw, "chunkset": chunkset, "ok": True, "timings": timings,
            "threshold": cal.get("threshold"),
            "n_chunks": len(det.get("chunks", [])),
            "n_fired": sum(1 for c in det.get("chunks", [])
                           if c.get("detections"))}


@app.function(image=image, volumes={"/data": data}, timeout=1800)
def compare_outcomes(root_a: str, root_b: str, keywords: list):
    """Threshold and per-chunk decision equality between two rebuild roots.

    fingerprint() covers the enrolment arrays; this covers what they produce.
    Two runs of a seeded pipeline should agree on both.
    """
    import sys
    sys.path.insert(0, "/app/core")
    from scoring import artifact_suffix
    suf = artifact_suffix()

    # Reuse modal_app.fingerprint's body rather than copying its hashing.
    # .local() runs the plain function inside this container; calling it as a
    # Modal function would fail, since it belongs to an app that is not running.
    from common.runtime import fingerprint
    prints = {"a": fingerprint.local(keywords, root_a),
              "b": fingerprint.local(keywords, root_b)}

    rows = []
    for kw in keywords:
        rec = {"keyword": kw}
        for tag, root in (("a", root_a), ("b", root_b)):
            cal_p = f"{root}/keywords/{kw}_calibration{suf}.json"
            det_p = f"{root}/logs/detections_{kw}.json"
            if not (os.path.exists(cal_p) and os.path.exists(det_p)):
                rec[tag] = None
                continue
            with open(cal_p) as f:
                cal = json.load(f)
            with open(det_p) as f:
                det = json.load(f)
            rec[tag] = {
                "threshold": cal["threshold"],
                "fired": sorted(c["file"] for c in det["chunks"]
                                if c.get("detections")),
                "above_thr": sorted(c["file"] for c in det["chunks"]
                                    if c["best_score"] >= det["threshold"]),
                "best": {c["file"]: c["best_score"] for c in det["chunks"]},
            }
        a, b = rec.get("a"), rec.get("b")
        if a and b:
            rec["threshold_equal"] = a["threshold"] == b["threshold"]
            rec["detector_equal"] = a["above_thr"] == b["above_thr"]
            rec["cascade_equal"] = a["fired"] == b["fired"]
            rec["max_abs_score_diff"] = max(
                abs(a["best"][f] - b["best"][f]) for f in a["best"])
        for tag in ("a", "b"):
            if rec.get(tag):
                rec[tag] = {"threshold": rec[tag]["threshold"],
                            "n_fired": len(rec[tag]["fired"])}
        rows.append(rec)
    return {"rows": rows, "fingerprints": prints}


@app.local_entrypoint()
def verify_determinism(root_a: str = REBUILD_ROOT,
                       root_b: str = "/data/run_rebuild2"):
    """Do two independent rebuilds agree, array for array and decision for
    decision? This is the other half of the determinism-fix claim."""
    kws = [kw for _, kw in EVAL]
    result = compare_outcomes.remote(root_a, root_b, kws)
    rows = result["rows"]
    fa, fb = result["fingerprints"]["a"], result["fingerprints"]["b"]

    same, differ, missing = [], [], []
    for key in sorted(set(fa) | set(fb)):
        x, y = fa.get(key), fb.get(key)
        if x is None or y is None:
            missing.append(key)
        elif x == y:
            same.append(key)
        else:
            differ.append((key, [k for k in x if x.get(k) != y.get(k)]))

    print(f"enrolment arrays: {len(same)} identical, {len(differ)} differing, "
          f"{len(missing)} missing")
    for key, arrays in differ:
        print(f"  DIFFERS {key}: {arrays}")
    for key in missing:
        print(f"  MISSING {key}")

    print()
    bad = 0
    for r in rows:
        ok = (r.get("threshold_equal") and r.get("detector_equal")
              and r.get("cascade_equal"))
        if not ok:
            bad += 1
        print(f"  {'ok  ' if ok else 'DIFF'} {r['keyword']:16s} "
              f"thr {r.get('a', {}).get('threshold')} vs "
              f"{r.get('b', {}).get('threshold')}  "
              f"max|score diff| = {r.get('max_abs_score_diff')}")
    print(f"\n{len(rows)} keywords, {bad} with any difference")
    print(json.dumps({"identical_arrays": len(same), "differing_arrays":
                      len(differ), "keywords_differing": bad}, indent=1))


@app.local_entrypoint()
def rebuild_all(root: str = REBUILD_ROOT):
    jobs = [{"keyword": kw, "chunkset": cs, "root": root} for cs, kw in EVAL]
    print(f"rebuilding {len(jobs)} keyword-by-set evaluations into {root}")
    rows, failed = [], []
    for r in rebuild.map(jobs, order_outputs=False):
        rows.append(r)
        status = "ok " if r.get("ok") else "FAIL"
        print(f"  {status} {r['chunkset']:12s} {r['keyword']:16s} "
              f"thr={r.get('threshold')} fired={r.get('n_fired')}/"
              f"{r.get('n_chunks')}")
        if not r.get("ok"):
            failed.append(r)
            print(f"       stage={r.get('stage')}: {str(r.get('err'))[:400]}")
    print(f"\n{len(rows)} jobs, {len(failed)} failed")
    totals = {}
    for r in rows:
        for k, v in (r.get("timings") or {}).items():
            totals[k] = totals.get(k, 0.0) + v
    for k, v in sorted(totals.items()):
        print(f"  {k:<20} {v / 3600:6.2f} core-hours  "
              f"({v / max(1, len(rows)):6.1f}s mean)")
    print(json.dumps(rows, indent=1))
