"""Transcribe live chunks with Whisper into a text document.

Output: <audio dir>/transcripts.txt (one block per chunk, in chunk order;
the audio dir follows SIAMESE_AUDIO_DIR).

Two callers:
  - Research: the CLI (`python transcribe_chunks.py`) transcribes a whole
    chunk set once, for ground truth (validate_detection.py) and
    calibrate.py's transcript leakage guard.
  - platform/live_worker.py: calls transcribe_chunks() directly, in-process
    (not as a subprocess), to transcribe only the bootstrap chunks as a
    one-time setup step, so the live platform's own calibration can use the
    same keyword_free_chunks() guard - see that module's docstring for why
    this replaced several transcription-free attempts that all had a
    failure mode (core/scoring.py's git history, commits
    28cffd5/bd0d441/18cd7b5, reverted). In-process because by this point
    live_worker.py has already imported torch/transformers/librosa (via
    scoring) - a subprocess would re-import all of that from scratch for
    no benefit (~40s measured on a CPU-only machine, more than a third of
    the whole 10-chunk transcription's wall time).

Default model is whisper-tiny, chosen for speed over -base/-small. Caveat:
a missed word here silently reintroduces the exact leakage this
transcription exists to prevent (a transcription false negative = a
keyword-bearing chunk wrongly treated as keyword-free) - tiny is the least
accurate option in the whisper family, and neither -base nor -small fully
resolved Whisper's occasional hallucination loop on noisy live audio
either (tested against 3 known-bad chunks with -small: 1 came out
correct, 2 still produced gibberish, just different gibberish), so this
is a deliberate speed-over-accuracy tradeoff, not a fix. --model can
override to any cached model (HF_HUB_OFFLINE is on by default - an
uncached model needs a first run with that disabled).
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
from scoring import AUDIO_DIR, PROJECT_ROOT, list_chunk_audios, spoken_numbers

warnings.filterwarnings("ignore")

# Whisper occasionally fails to predict an end-of-transcript token on a
# ~5s chunk and instead loops on a repeated (often non-English) phrase for
# its full generation budget - observed live: 3 of 10 bootstrap chunks each
# took 58-74s (vs ~6-9s normal) and produced garbage like "Mae'r gweithio'r
# gweithio'r gweithio'r ...". no_repeat_ngram_size forces the decoder off
# the loop; max_new_tokens caps the worst case regardless (a real ~5s
# chunk's transcript never gets close to 128 tokens - see the normal chunks
# above, all well under 20 words). This bounds the pathological latency but
# does NOT fully fix the underlying hallucination - a looping chunk still
# comes out as gibberish, just quickly, so keyword_free_chunks() can still
# misjudge it; see reports/EXPERIMENT_LOG.md if that recurs.
GENERATE_KWARGS = {"no_repeat_ngram_size": 3, "max_new_tokens": 128}


def transcribe_chunks(out_path, model="openai/whisper-tiny", limit=None):
    """Transcribe chunks in AUDIO_DIR (or just the first `limit`) to out_path.

    Returns the number of chunks transcribed (0 if none were found).
    """
    t0 = time.perf_counter()
    ui.banner("CHUNK TRANSCRIPTION", model.rsplit("/", 1)[-1])

    audio_files = list_chunk_audios()
    if limit is not None:
        audio_files = audio_files[:limit]
    if not audio_files:
        ui.fail(f"No chunks in {AUDIO_DIR} - run downloader.py first.")
        return 0

    ui.step(f"loading {model} ...")
    import torch
    # No GPU on the target machines (torch.cuda.is_available() is False) -
    # torch's own CPU default (4 intra-op threads, observed) leaves half of
    # an 8-core machine idle during the ASR forward pass, which is the only
    # CPU-heavy work happening at this specific point in setup.
    torch.set_num_threads(os.cpu_count())
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model=model)

    lines = [f"Chunk transcripts - generated {datetime.now():%Y-%m-%d %H:%M:%S}",
             f"Source directory: {AUDIO_DIR} ({len(audio_files)} chunks)", ""]
    for audio_file in audio_files:
        name = os.path.basename(audio_file)
        text = spoken_numbers(
            asr(audio_file, generate_kwargs=GENERATE_KWARGS)["text"].strip())
        preview = (text[:52] + "...") if len(text) > 55 else text
        ui.item(f"{name:<14} {preview if text else '(no speech detected)'}")
        lines.append(f"[{name}]")
        lines.append(text if text else "(no speech detected)")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    ui.ok(f"{len(audio_files)} chunks transcribed")
    ui.done(os.path.relpath(out_path, PROJECT_ROOT), t0)
    return len(audio_files)


def main():
    ap = argparse.ArgumentParser(description="Write chunk transcripts to a text file.")
    ap.add_argument("--out", default=os.path.join(AUDIO_DIR, "transcripts.txt"))
    ap.add_argument("--model", default="openai/whisper-tiny",
                    help="HF model id; must be locally cached under "
                         "offline mode (default HF_HUB_OFFLINE=1)")
    ap.add_argument("--limit", type=int, default=None,
                    help="transcribe only the first N chunks (by index) "
                         "instead of every chunk in the audio dir")
    args = ap.parse_args()
    transcribe_chunks(args.out, model=args.model, limit=args.limit)


if __name__ == "__main__":
    main()
