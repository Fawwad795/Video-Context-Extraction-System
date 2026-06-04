"""VMS-integrated Siamese keyword detector (roadmap phase P4).

Parallel to Stream*_corelation_updated_v2.py: same VMS folder layout and outputs (per-keyword
timestamp log + a saved +/-5-chunk context clip), but matching uses the trained Siamese model
(embed -> cosine vs the synthesized-keyword prototype) instead of np.correlate + a 70% cutoff.

Invoked per stream by the thin wrappers Stream1_siamese_detect.py / Stream2_siamese_detect.py,
which the GUI launches when the environment variable VMS_DETECTOR=siamese.

Config via env vars:
  VMS_SIAMESE_CKPT       path to the trained checkpoint (default: ../siamese/checkpoints/siamese_full_v1.pt)
  VMS_SIAMESE_THRESHOLD  cosine cutoff for a detection (default: 0.80)
  VMS_SIAMESE_DEVICE     'cpu' or 'cuda' (default: cpu)
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import sys
import time
from datetime import datetime

# Make the repo-root `siamese` package importable when run from Research/.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
from siamese.detector import SiameseDetector  # noqa: E402

BASE = os.path.expanduser(os.path.join("~", "VMS", "GUI2CHjetson"))
DEFAULT_CKPT = os.path.join(_REPO, "siamese", "checkpoints", "siamese_full_v1.pt")


def _keywords_in(searchword_dir: str) -> list[str]:
    """Distinct keywords present as synthesized clips named '{speaker}-{word}.wav'."""
    words = set()
    for p in glob.glob(os.path.join(searchword_dir, "*.wav")):
        words.add(os.path.basename(p).split("-", 1)[-1].rsplit(".", 1)[0])
    return sorted(words)


def _audio_index(fname: str):
    m = re.search(r"live_(\d+)\.wav", fname)
    return int(m.group(1)) if m else None


def _save_detection(word: str, idx: int, score: float, video_dir: str, detect_dir: str):
    """Write the timestamp log and concatenate the +/-5-chunk context clip (as the original does)."""
    out_dir = os.path.join(detect_dir, word)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(out_dir, "timestamps.txt"), "a") as f:
        f.write(f"[{ts}] Detected '{word}' (score {score:.3f}) in live_{idx}.mp4\n")
    print(f"[siamese] DETECTED '{word}' score {score:.3f} in live_{idx}.mp4", flush=True)

    out_clip = os.path.join(out_dir, f"{word}_detected_at_{idx}.mp4")
    with open(out_clip, "ab") as outfile:
        for i in range(max(0, idx - 5), idx + 6):
            v = os.path.join(video_dir, f"live_{i}.mp4")
            if os.path.exists(v):
                with open(v, "rb") as infile:
                    shutil.copyfileobj(infile, outfile)


def run(stream_name: str):
    ckpt = os.environ.get("VMS_SIAMESE_CKPT", DEFAULT_CKPT)
    threshold = float(os.environ.get("VMS_SIAMESE_THRESHOLD", "0.80"))
    device = os.environ.get("VMS_SIAMESE_DEVICE", "cpu")
    searchword_dir = os.path.join(BASE, f"{stream_name}_searchword1")
    audio_dir = os.path.join(BASE, f"{stream_name}audios")
    video_dir = os.path.join(BASE, f"{stream_name}videos")
    detect_dir = os.path.join(BASE, f"{stream_name}_detection")

    if not os.path.exists(ckpt):
        print(f"[siamese] checkpoint not found: {ckpt}", flush=True)
        return
    print(f"[siamese] {stream_name}: loading {os.path.basename(ckpt)} "
          f"(device={device}, threshold={threshold})", flush=True)
    det = SiameseDetector(ckpt, device=device, threshold=threshold)

    # Wait for the synthesized keyword clips, then build one prototype per keyword.
    print("[siamese] waiting for synthesized keyword clips...", flush=True)
    while True:
        keywords = _keywords_in(searchword_dir) if os.path.isdir(searchword_dir) else []
        if keywords:
            break
        time.sleep(2)
    prototypes = {kw: det.build_prototype(searchword_dir, keyword=kw)[0] for kw in keywords}
    print(f"[siamese] prototypes built for: {list(prototypes)}", flush=True)

    # Poll the audio folder and score each new chunk against every keyword prototype.
    processed: set[int] = set()
    print("[siamese] monitoring live audio chunks...", flush=True)
    while True:
        if os.path.isdir(audio_dir):
            files = sorted(
                (f for f in os.listdir(audio_dir) if f.endswith(".wav")),
                key=lambda f: _audio_index(f) or 0,
            )
            for af in files:
                idx = _audio_index(af)
                if idx is None or idx in processed:
                    continue
                wav_path = os.path.join(audio_dir, af)
                best_kw, best_score = None, -2.0
                for kw, proto in prototypes.items():
                    det.prototype, det.keyword = proto, kw
                    try:
                        s = det.score_file(wav_path)
                    except Exception as e:
                        print(f"[siamese] error scoring {af}: {e}", flush=True)
                        s = -2.0
                    if s > best_score:
                        best_kw, best_score = kw, s
                processed.add(idx)
                if best_score >= threshold:
                    _save_detection(best_kw, idx, best_score, video_dir, detect_dir)
                else:
                    print(f"[siamese] live_{idx}: best '{best_kw}' {best_score:.3f} (no match)",
                          flush=True)
        time.sleep(3)
