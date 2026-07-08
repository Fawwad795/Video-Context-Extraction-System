"""Phoneme verification of detector output (offline pipeline, precision stage).

Re-checks every chunk in logs/detections_<keyword>.json through the phone
view (core/phoneme_verify.py): a chunk's detections survive only if the
top-scoring windows actually decode to the keyword's phones. Chunks that
fail are rewritten with empty detections; the original file is backed up as
*_unverified.json. Run validate_detection.py afterwards for verified P/R/F1.

References and the accept threshold tau are calibrated once per keyword and
cached in keywords/<keyword>_phone_cache.json (delete it to recalibrate).

Usage: python pipeline/verify_detections.py --keyword party
"""

import argparse
import json
import os
import shutil
import time

import numpy as np

# Shared modules live in ../core
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "core"))

import console as ui
import phoneme_verify as pv
from scoring import PROJECT_ROOT, SAMPLE_RATE, keyword_free_chunks

if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser(description="Phoneme verification of detections.")
    ap.add_argument("--keyword", default=None, help="defaults to selected_keyword.txt")
    ap.add_argument("--tau", type=float, default=None,
                    help="accept threshold override; default = cached/calibrated")
    ap.add_argument("--max-refs", type=int, default=8)
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
    ui.banner("PHONEME VERIFICATION", f"keyword: {keyword}")

    json_path = os.path.join(PROJECT_ROOT, "logs", f"detections_{keyword}.json")
    if not os.path.exists(json_path):
        ui.fail(f"{json_path} not found - run detector.py first.")
        return
    with open(json_path) as f:
        results = json.load(f)
    window_seconds = float(results["window_seconds"])

    ui.step(f"loading phoneme recognizer {pv.PHONEME_MODEL} ...")
    processor, model, torch = pv.load_phoneme_model()

    cache_path = os.path.join(PROJECT_ROOT, "keywords",
                              f"{keyword}_phone_cache.json")
    cached = pv.load_phone_cache(cache_path)
    if cached and args.tau is None:
        refs, tau = cached
        ui.ok(f"refs + tau loaded from cache ({len(refs)} refs, tau={tau:.3f})")
    else:
        ui.step("building phoneme references from anchor variants ...")
        variants_dir = os.path.join(PROJECT_ROOT, "keywords", f"{keyword}_variants")
        refs, loo_mean = pv.build_references(
            keyword, variants_dir, processor, model, torch,
            max_refs=args.max_refs, log=ui.item)
        if not refs:
            ui.fail("No usable reference decodes - aborting.")
            return
        ui.kv("reference self-similarity", f"{loo_mean:.3f}")
        if args.tau is not None:
            tau = args.tau
        else:
            ui.step("calibrating tau on keyword-free stream windows ...")
            rng = np.random.default_rng(args.seed)
            tau = pv.calibrate_tau(refs, loo_mean, window_seconds,
                                   keyword_free_chunks(keyword), processor,
                                   model, torch, rng, log=ui.item)
            pv.save_phone_cache(cache_path, keyword, refs, tau, loo_mean)
            ui.ok(f"cached refs + tau -> {os.path.relpath(cache_path, PROJECT_ROOT)}")
    ui.kv("accept threshold tau", f"{tau:.3f}")

    import librosa
    from scoring import AUDIO_DIR
    kept = dropped = 0
    for chunk in results["chunks"]:
        dets = chunk.get("detections") or []
        if not dets:
            continue
        y, _ = librosa.load(os.path.join(AUDIO_DIR, chunk["file"]),
                            sr=SAMPLE_RATE)
        ok, sim = pv.verify_chunk(y, dets, window_seconds, refs, tau,
                                  processor, model, torch)
        chunk["phone_verified"] = ok
        chunk["best_phone_sim"] = round(sim, 3)
        if ok:
            kept += 1
            ui.ok(f"{chunk['file']}: verified (phone-sim {sim:.2f}, "
                  f"{len(dets)} windows)")
        else:
            dropped += 1
            chunk["detections"] = []
            ui.warn(f"{chunk['file']}: rejected (phone-sim {sim:.2f} < "
                    f"tau {tau:.2f}) - detections cleared")

    results["phone_verification"] = {
        "model": pv.PHONEME_MODEL, "tau": tau, "n_refs": len(refs),
        "decode_span_s": pv.DECODE_SPAN_S,
    }
    backup = json_path.replace(".json", "_unverified.json")
    if not os.path.exists(backup):
        shutil.copy2(json_path, backup)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    ui.kv("chunks", f"{kept} verified, {dropped} rejected")
    ui.done(os.path.relpath(json_path, PROJECT_ROOT), t0)
    ui.item(f"next: python pipeline/validate_detection.py --keyword {keyword}")


if __name__ == "__main__":
    main()
