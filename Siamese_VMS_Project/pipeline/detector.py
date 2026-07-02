"""Live keyword detection with AS-norm adaptive scoring (Phase 1).

What changed vs. the original detector:
  anchor    - multi-voice TTS prototype centroid (keywords/<kw>_anchor.npz)
              instead of a single TTS/human clip (override: --anchor-audio);
  score     - cosine similarity on L2-normalized embeddings instead of raw
              L2 distance;
  decision  - Adaptive S-norm against an impostor cohort, compared to the
              calibrated per-keyword threshold from calibrate.py, instead
              of a hardcoded distance like 1.25;
  speed     - sliding windows are embedded in batches.

Outputs: logs/timestamps_<keyword>.txt   (append, human-readable)
         logs/detections_<keyword>.json  (overwrite, for validate_detection.py)
"""

import argparse
import json
import os
from datetime import datetime

import librosa
import numpy as np

# Shared modules (scoring, siamese_model, augment_utils) live in ../core
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "core"))

from scoring import (DEFAULT_TOP_K, PROJECT_ROOT, SAMPLE_RATE, asnorm_windows,
                     embed_batch, l2_normalize, list_chunk_audios,
                     load_cohort, load_siamese_model)

DEFAULT_FALLBACK_THRESHOLD = 2.5


def load_anchor(keyword, anchor_audio, model):
    """Returns (anchor_embedding [D], window_samples, description)."""
    if anchor_audio:
        y, _ = librosa.load(anchor_audio, sr=SAMPLE_RATE)
        y, _ = librosa.effects.trim(y, top_db=30)
        emb = embed_batch(model, [y.astype(np.float32)])[0]
        return emb, len(y), f"audio file {os.path.basename(anchor_audio)}"

    npz_path = os.path.join(PROJECT_ROOT, "keywords", f"{keyword}_anchor.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"{npz_path} not found - run keyword_generator.py first "
            f"(or pass --anchor-audio <wav/m4a> to use a recorded anchor).")
    data = np.load(npz_path)
    return (l2_normalize(data["centroid"]), int(data["window_samples"]),
            "TTS prototype centroid")


def resolve_threshold(keyword, override):
    if override is not None:
        return float(override), "command-line override"
    calib_path = os.path.join(PROJECT_ROOT, "keywords", f"{keyword}_calibration.json")
    if os.path.exists(calib_path):
        with open(calib_path) as f:
            calib = json.load(f)
        return float(calib["threshold"]), f"calibrated ({calib_path})"
    print(f"WARNING: no calibration found - using fallback threshold "
          f"{DEFAULT_FALLBACK_THRESHOLD}. Run calibrate.py for a fitted one.")
    return DEFAULT_FALLBACK_THRESHOLD, "uncalibrated fallback"


def run_detection(keyword, anchor_audio=None, threshold=None, step_seconds=0.05,
                  top_k=DEFAULT_TOP_K, batch_size=16, scales=(0.6, 0.8, 1.0),
                  top_candidates=8):
    print("Initializing Siamese AS-norm detector...")
    model = load_siamese_model()
    anchor, window_samples, anchor_desc = load_anchor(keyword, anchor_audio, model)
    cohort = load_cohort(keyword)
    threshold, threshold_desc = resolve_threshold(keyword, threshold)

    audio_files = list_chunk_audios()
    if not audio_files:
        print("No live chunks found in audios/ - run downloader.py first.")
        return

    step_samples = max(1, int(SAMPLE_RATE * step_seconds))
    log_path = os.path.join(PROJECT_ROOT, "logs", f"timestamps_{keyword}.txt")
    json_path = os.path.join(PROJECT_ROOT, "logs", f"detections_{keyword}.json")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    print(f"\nKeyword: '{keyword}'  |  anchor: {anchor_desc}")
    print(f"Window: {window_samples / SAMPLE_RATE:.2f}s x scales {list(scales)} "
          f"(humans often speak faster than TTS)  step: {step_seconds * 1000:.0f}ms  "
          f"cohort: {cohort.shape[0]}  top-k: {top_k}")
    print(f"Threshold: {threshold:.3f} AS-norm units ({threshold_desc})")
    print(f"Scanning {len(audio_files)} chunks...\n")

    results = {"keyword": keyword, "threshold": threshold, "anchor": anchor_desc,
               "window_seconds": window_samples / SAMPLE_RATE, "chunks": []}

    for audio_file in audio_files:
        filename = os.path.basename(audio_file)
        y, _ = librosa.load(audio_file, sr=SAMPLE_RATE)

        # Multi-scale scan: a human saying the keyword faster/slower than the
        # TTS anchor still gets a window that fits the spoken duration.
        all_scores, all_raw, all_times, all_scales, n_windows = [], [], [], [], 0
        for scale in scales:
            ws = max(int(window_samples * scale), int(0.15 * SAMPLE_RATE))
            starts = list(range(0, len(y) - ws + 1, step_samples))
            if not starts:
                continue
            windows = [y[s:s + ws].astype(np.float32) for s in starts]
            embs = embed_batch(model, windows, batch_size=batch_size)
            normed, raw = asnorm_windows(embs, anchor, cohort, top_k)
            all_scores.append(normed)
            all_raw.append(raw)
            all_times.append(np.array(starts) / SAMPLE_RATE)
            all_scales.append(np.full(len(starts), scale))
            n_windows += len(starts)

        if n_windows == 0:
            print(f"{filename}: shorter than the keyword window - skipped.")
            results["chunks"].append(
                {"file": filename, "skipped": True, "detections": []})
            continue

        normed = np.concatenate(all_scores)
        raw = np.concatenate(all_raw)
        times = np.concatenate(all_times)
        win_scales = np.concatenate(all_scales)

        best = int(np.argmax(normed))
        hit_idx = np.where(normed >= threshold)[0]
        detections = [{"time": float(times[i]),
                       "score": float(normed[i]),
                       "raw_cos": float(raw[i]),
                       "scale": float(win_scales[i])} for i in hit_idx]

        # Top-N windows regardless of threshold (greedy NMS: skip windows
        # within 0.25s of an already-kept higher-scoring one). A second-stage
        # verifier (verify_detections.py) can rescue a true keyword window
        # whose embedding score fell just below the threshold - stage 1
        # supplies recall candidates, stage 2 enforces precision.
        candidates = []
        for i in np.argsort(-normed):
            t = float(times[i])
            if any(abs(t - c["time"]) < 0.25 for c in candidates):
                continue
            candidates.append({"time": t,
                               "score": float(normed[i]),
                               "raw_cos": float(raw[i]),
                               "scale": float(win_scales[i])})
            if len(candidates) >= top_candidates:
                break

        chunk_record = {"file": filename,
                        "n_windows": n_windows,
                        "best_score": float(normed[best]),
                        "best_raw_cos": float(raw[best]),
                        "best_time": float(times[best]),
                        "best_scale": float(win_scales[best]),
                        "detections": detections,
                        "candidates": candidates}
        results["chunks"].append(chunk_record)

        if len(hit_idx) > 0:
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            msg = (f"[{stamp}] Match found! AS-norm score: {normed[best]:.2f} "
                   f"(cos {raw[best]:.3f}, threshold {threshold:.2f}, "
                   f"scale {win_scales[best]:.1f}x) | "
                   f"Chunk: {filename} at {times[best]:.1f}s "
                   f"({len(hit_idx)} windows above threshold)")
            print(msg)
            with open(log_path, "a") as f:
                f.write(msg + "\n")
        else:
            print(f"Processed {filename} - no match. "
                  f"(best AS-norm {normed[best]:.2f} at {times[best]:.1f}s, "
                  f"cos {raw[best]:.3f}, scale {win_scales[best]:.1f}x)")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    n_hits = sum(1 for c in results["chunks"] if c.get("detections"))
    print(f"\nDone. {n_hits}/{len(audio_files)} chunks contained detections.")
    print(f"Results: {json_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AS-norm keyword detector.")
    ap.add_argument("--keyword", default=None, help="defaults to selected_keyword.txt")
    ap.add_argument("--anchor-audio", default=None,
                    help="optional recorded anchor (wav/m4a) instead of the TTS centroid")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the calibrated AS-norm threshold")
    ap.add_argument("--step", type=float, default=0.05, help="window hop in seconds")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--scales", default="0.6,0.8,1.0",
                    help="comma-separated window scales relative to the anchor duration")
    ap.add_argument("--top-candidates", type=int, default=8,
                    help="top-N NMS windows per chunk saved for the verifier")
    args = ap.parse_args()

    keyword = args.keyword
    if keyword is None:
        kw_file = os.path.join(PROJECT_ROOT, "selected_keyword.txt")
        if not os.path.exists(kw_file):
            print("No --keyword given and selected_keyword.txt not found. Run transcriber.py first.")
            raise SystemExit(1)
        keyword = open(kw_file).read().strip()

    run_detection(keyword, anchor_audio=args.anchor_audio, threshold=args.threshold,
                  step_seconds=args.step, top_k=args.top_k, batch_size=args.batch_size,
                  scales=tuple(float(s) for s in args.scales.split(",")),
                  top_candidates=args.top_candidates)
