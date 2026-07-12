"""Verification of detector output (offline pipeline, precision stage).

Re-checks every chunk in logs/detections_<keyword>.json through rival-anchor
verification (core/rival_verify.py) before the numbers are trusted: the
keyword's synthesized phonetic near-neighbours act as explicit impostor
anchors, and a detection survives only if its window embeddings beat every
rival by the calibrated margin delta. Requires
keywords/<kw>_rivals<suffix>.npz (built automatically via
pipeline/rival_builder.py when missing). The retired phone-CTC stage is
preserved on the archive/phone-verifier branch.

Chunks that fail are rewritten with empty detections; the original file is
backed up as *_unverified.json. Run validate_detection.py afterwards for
verified P/R/F1.

Usage: python pipeline/verify_detections.py --keyword party
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

# Shared modules live in ../core
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "core"))

import console as ui
from scoring import PROJECT_ROOT, SAMPLE_RATE

if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _load_results(keyword):
    json_path = os.path.join(PROJECT_ROOT, "logs", f"detections_{keyword}.json")
    if not os.path.exists(json_path):
        ui.fail(f"{json_path} not found - run detector.py first.")
        return None, None
    with open(json_path) as f:
        return json_path, json.load(f)


def _write_results(json_path, results, kept, dropped, t0):
    backup = json_path.replace(".json", "_unverified.json")
    if not os.path.exists(backup):
        shutil.copy2(json_path, backup)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    ui.kv("chunks", f"{kept} verified, {dropped} rejected")
    ui.done(os.path.relpath(json_path, PROJECT_ROOT), t0)
    ui.item(f"next: python pipeline/validate_detection.py --keyword "
            f"{results['keyword']}")


# --------------------------------------------------------------------------
# Stage: rival-anchor verification (default)
# --------------------------------------------------------------------------

def verify_rival(keyword, args):
    import rival_verify as rv
    from scoring import anchor_path, load_cohort, load_siamese_model

    t0 = time.perf_counter()
    ui.banner("RIVAL-ANCHOR VERIFICATION", f"keyword: {keyword}")
    json_path, results = _load_results(keyword)
    if results is None:
        return

    npz_path = rv.rivals_path(keyword, PROJECT_ROOT)
    if not os.path.exists(npz_path):
        ui.step("rival anchors missing - building (one-time) ...")
        r = subprocess.run([sys.executable,
                            os.path.join(os.path.dirname(__file__),
                                         "rival_builder.py"),
                            "--keyword", keyword])
        if r.returncode != 0 or not os.path.exists(npz_path):
            ui.fail("rival_builder failed - aborting.")
            return
    words, centroids, delta = rv.load_rivals(npz_path)
    if args.delta is not None:
        delta = args.delta
    ui.kv("rivals", ", ".join(words))
    ui.kv("accept margin delta", f"{delta:+.3f}")

    z = np.load(anchor_path(keyword))
    anchor = z["centroid"]
    window_samples = int(z["window_samples"])
    try:
        cohort = load_cohort(keyword)
    except Exception:
        cohort = None
        ui.warn("no cohort - raw-cosine margins")
    ui.step("loading embedding model ...")
    model = load_siamese_model()

    import librosa
    from scoring import AUDIO_DIR
    kept = dropped = 0
    for chunk in results["chunks"]:
        dets = chunk.get("detections") or []
        if not dets:
            continue
        y, _ = librosa.load(os.path.join(AUDIO_DIR, chunk["file"]),
                            sr=SAMPLE_RATE)
        ok, margin, rival = rv.verify_chunk(
            y, dets, window_samples, model, anchor, words, centroids,
            delta, cohort=cohort)
        chunk["rival_verified"] = ok
        chunk["best_margin"] = round(margin, 4)
        chunk["nearest_rival"] = rival
        if ok:
            kept += 1
            ui.ok(f"{chunk['file']}: verified (margin {margin:+.3f} vs "
                  f"'{rival}', {len(dets)} windows)")
        else:
            dropped += 1
            chunk["detections"] = []
            ui.warn(f"{chunk['file']}: rejected (margin {margin:+.3f} < "
                    f"delta {delta:+.3f}; nearest rival '{rival}') - "
                    f"detections cleared")

    results["rival_verification"] = {
        "rivals": words, "delta": delta, "n_rivals": len(words),
    }
    _write_results(json_path, results, kept, dropped, t0)


def main():
    ap = argparse.ArgumentParser(description="Verification of detections.")
    ap.add_argument("--keyword", default=None, help="defaults to selected_keyword.txt")
    ap.add_argument("--delta", type=float, default=None,
                    help="rival accept margin override; default = calibrated")
    args = ap.parse_args()

    keyword = args.keyword
    if keyword is None:
        kw_file = os.path.join(PROJECT_ROOT, "selected_keyword.txt")
        if not os.path.exists(kw_file):
            ui.fail("No --keyword given and selected_keyword.txt not found.")
            return
        keyword = open(kw_file).read().strip()
    keyword = keyword.lower()

    verify_rival(keyword, args)


if __name__ == "__main__":
    main()
