"""Transcribe every live chunk with whisper-tiny into a text document.

Output: transcripts.txt (one block per chunk, in chunk order).
"""

import argparse
import os
import warnings
from datetime import datetime

from scoring import PROJECT_ROOT, list_chunk_audios

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser(description="Write chunk transcripts to a text file.")
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "transcripts.txt"))
    args = ap.parse_args()

    audio_files = list_chunk_audios()
    if not audio_files:
        print("No chunks in audios/ - run downloader.py first.")
        return

    print("Loading whisper-tiny...")
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

    lines = [f"Chunk transcripts - generated {datetime.now():%Y-%m-%d %H:%M:%S}",
             f"Source directory: audios/ ({len(audio_files)} chunks)", ""]
    for audio_file in audio_files:
        name = os.path.basename(audio_file)
        print(f"Transcribing {name}...")
        text = asr(audio_file)["text"].strip()
        lines.append(f"[{name}]")
        lines.append(text if text else "(no speech detected)")
        lines.append("")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Transcripts written to {args.out}")


if __name__ == "__main__":
    main()
