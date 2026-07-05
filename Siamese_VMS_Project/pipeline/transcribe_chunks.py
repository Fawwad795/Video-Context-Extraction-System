"""Transcribe live chunks with Whisper into a text document.

Output: <audio dir>/transcripts.txt (one block per chunk, in chunk order;
the audio dir follows SIAMESE_AUDIO_DIR).

Two callers:
  - Research: transcribe a whole chunk set once, for ground truth
    (validate_detection.py) and calibrate.py's transcript leakage guard.
  - platform/live_worker.py: transcribe only the bootstrap chunks
    (--limit) as a one-time setup step, so the live platform's own
    calibration can use the same keyword_free_chunks() guard - see that
    module's docstring for why this replaced several transcription-free
    attempts that all had a failure mode (core/scoring.py's git history,
    commits 28cffd5/bd0d441/18cd7b5, reverted).

Default model is whisper-base, not -tiny: a missed word here silently
reintroduces the exact leakage this transcription exists to prevent (a
transcription false negative = a keyword-bearing chunk wrongly treated as
keyword-free), so accuracy matters more than for ground-truth evaluation,
where a few noisy transcripts average out. Base is already cached locally
and only ~2x tiny's params; --model can override to any cached model
(HF_HUB_OFFLINE is on by default - an uncached model needs a first run
with that disabled).
"""

import argparse
import os
import time
import warnings
from datetime import datetime

# Shared modules (scoring, siamese_model, augment_utils) live in ../core
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "core"))

import console as ui
from scoring import AUDIO_DIR, PROJECT_ROOT, list_chunk_audios

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser(description="Write chunk transcripts to a text file.")
    ap.add_argument("--out", default=os.path.join(AUDIO_DIR, "transcripts.txt"))
    ap.add_argument("--model", default="openai/whisper-base",
                    help="HF model id; must be locally cached under "
                         "offline mode (default HF_HUB_OFFLINE=1)")
    ap.add_argument("--limit", type=int, default=None,
                    help="transcribe only the first N chunks (by index) "
                         "instead of every chunk in the audio dir")
    args = ap.parse_args()

    t0 = time.perf_counter()
    ui.banner("CHUNK TRANSCRIPTION", args.model.rsplit("/", 1)[-1])

    audio_files = list_chunk_audios()
    if args.limit is not None:
        audio_files = audio_files[:args.limit]
    if not audio_files:
        ui.fail(f"No chunks in {AUDIO_DIR} - run downloader.py first.")
        return

    ui.step(f"loading {args.model} ...")
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model=args.model)

    lines = [f"Chunk transcripts - generated {datetime.now():%Y-%m-%d %H:%M:%S}",
             f"Source directory: {AUDIO_DIR} ({len(audio_files)} chunks)", ""]
    for audio_file in audio_files:
        name = os.path.basename(audio_file)
        text = asr(audio_file)["text"].strip()
        preview = (text[:52] + "...") if len(text) > 55 else text
        ui.item(f"{name:<14} {preview if text else '(no speech detected)'}")
        lines.append(f"[{name}]")
        lines.append(text if text else "(no speech detected)")
        lines.append("")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    ui.ok(f"{len(audio_files)} chunks transcribed")
    ui.done(os.path.relpath(args.out, PROJECT_ROOT), t0)


if __name__ == "__main__":
    main()
