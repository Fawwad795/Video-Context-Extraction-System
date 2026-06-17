"""Transcribe chunks with whisper-tiny -> a text file (ground-truth reference).

Self-contained (no project imports). Use the transcripts to pick a keyword and
note which chunks contain it, then pass those to detect.py --ground-truth.

    python transcribe_chunks.py                              # ./new_chunks/audios
    python transcribe_chunks.py --audio-dir <dir> --out transcripts.txt

Run with a Python that has transformers + torch (see requirements-infra.txt).
"""
import argparse
import glob
import os
import re
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))


def list_chunks(audio_dir):
    files = glob.glob(os.path.join(audio_dir, "*.wav"))

    def order(f):
        m = re.search(r"(\d+)", os.path.basename(f))
        return int(m.group(1)) if m else 0

    return sorted(files, key=order)


def main():
    ap = argparse.ArgumentParser(description="Transcribe chunks with whisper-tiny.")
    ap.add_argument("--audio-dir", default=os.path.join(HERE, "new_chunks", "audios"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(os.path.dirname(args.audio_dir), "transcripts.txt")

    files = list_chunks(args.audio_dir)
    if not files:
        print(f"No .wav files in {args.audio_dir}")
        return

    print("Loading whisper-tiny...")
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

    lines = [f"Chunk transcripts - generated {datetime.now():%Y-%m-%d %H:%M:%S}",
             f"Source: {args.audio_dir} ({len(files)} chunks)", ""]
    for f in files:
        name = os.path.basename(f)
        print(f"Transcribing {name}...")
        text = asr(f)["text"].strip()
        lines += [f"[{name}]", text if text else "(no speech detected)", ""]

    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"Transcripts written to {out}")


if __name__ == "__main__":
    main()
