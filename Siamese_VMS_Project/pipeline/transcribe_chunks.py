"""Transcribe every live chunk with whisper-tiny into a text document.

Output: <audio dir>/transcripts.txt (one block per chunk, in chunk order;
the audio dir follows SIAMESE_AUDIO_DIR).
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
    args = ap.parse_args()

    t0 = time.perf_counter()
    ui.banner("CHUNK TRANSCRIPTION", "whisper-tiny")

    audio_files = list_chunk_audios()
    if not audio_files:
        ui.fail(f"No chunks in {AUDIO_DIR} - run downloader.py first.")
        return

    ui.step("loading whisper-tiny ...")
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

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
