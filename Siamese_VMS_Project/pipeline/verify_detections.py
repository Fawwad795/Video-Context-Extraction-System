"""Verification of detector output (offline pipeline, precision stage).

Re-checks every chunk in logs/detections_<keyword>.json through a second
view before the numbers are trusted. Two stages are available, selected by
SIAMESE_VERIFIER (or --stage):

  rival  (default) - rival-anchor verification (core/rival_verify.py): the
          keyword's synthesized phonetic near-neighbours act as explicit
          impostor anchors; a detection survives only if its window
          embeddings beat every rival by the calibrated margin delta.
          Requires keywords/<kw>_rivals<suffix>.npz (built automatically via
          pipeline/rival_builder.py when missing).
  phone  - the retired phone-sequence stage (core/phoneme_verify.py), kept
          for ablation: wide-span CTC decode matched against LOO-filtered
          TTS reference phone strings, threshold tau cached per keyword.

Chunks that fail are rewritten with empty detections; the original file is
backed up as *_unverified.json. Run validate_detection.py afterwards for
verified P/R/F1.

Usage: python pipeline/verify_detections.py --keyword party [--stage rival]
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
from scoring import PROJECT_ROOT, SAMPLE_RATE, keyword_free_chunks

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


# --------------------------------------------------------------------------
# Stage: phone-sequence verification (retired; kept for ablation)
# --------------------------------------------------------------------------

def verify_phone(keyword, args):
    import phoneme_verify as pv

    t0 = time.perf_counter()
    ui.banner("PHONEME VERIFICATION", f"keyword: {keyword}")
    json_path, results = _load_results(keyword)
    if results is None:
        return
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
    _write_results(json_path, results, kept, dropped, t0)


def main():
    ap = argparse.ArgumentParser(description="Verification of detections.")
    ap.add_argument("--keyword", default=None, help="defaults to selected_keyword.txt")
    ap.add_argument("--stage", default=None, choices=["rival", "phone"],
                    help="verifier stage; default = $SIAMESE_VERIFIER or 'rival'")
    ap.add_argument("--delta", type=float, default=None,
                    help="rival accept margin override; default = calibrated")
    ap.add_argument("--tau", type=float, default=None,
                    help="phone accept threshold override; default = cached")
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
    keyword = keyword.lower()

    stage = args.stage or os.environ.get("SIAMESE_VERIFIER", "rival")
    if stage == "phone":
        verify_phone(keyword, args)
    else:
        verify_rival(keyword, args)


if __name__ == "__main__":
    main()
