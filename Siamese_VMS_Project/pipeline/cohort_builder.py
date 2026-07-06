"""Build the AS-norm impostor cohort.

The cohort answers "how close does a *non*-keyword typically get to the
anchor / to a window?" - every detection score is normalized against it,
which is what replaces the hardcoded distance threshold.

Two impostor sources:
  1. random keyword-length windows sampled from the live chunks -
     domain-matched negatives (speech, music, silence of the actual stream),
     excluding any chunk whose transcript contains the keyword or a stem
     derivative (the same keyword_free_chunks guard calibrate.py uses), so a
     real keyword utterance never lands in the cohort the detector normalizes
     against;
  2. TTS distractor words in random voices - same-domain competition for the
     TTS anchor, so the anchor's synthetic-domain advantage is normalized
     away. Off by default: the pipeline ablation study (ablation_study/)
     found no F1 gain from TTS distractors with the trained wavlm head;
     enable with --tts to add them back.

Output: cohort.npz (embeddings [N, D])
"""

import argparse
import os
import time

import librosa
import numpy as np

# Shared modules (scoring, siamese_model, augment_utils) live in ../core
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "core"))

import console as ui
from scoring import (PROJECT_ROOT, SAMPLE_RATE, anchor_path, cohort_path,
                     embed_batch, keyword_free_chunks, load_siamese_model,
                     sample_stream_windows)

DISTRACTOR_WORDS = [
    "people", "because", "through", "before", "little", "world", "right",
    "think", "never", "again", "great", "house", "water", "sound", "place",
    "years", "being", "every", "thought", "really", "always", "together",
    "important", "question", "government", "different", "number", "example",
    "money", "music", "morning", "nothing", "problem", "country", "actually",
    "probably", "understand", "information", "interest", "history", "moment",
    "minute", "percent", "weather", "police", "market", "report", "support",
    "control", "change",
]


def main():
    ap = argparse.ArgumentParser(description="Build the AS-norm impostor cohort.")
    ap.add_argument("--keyword", default=None, help="defaults to selected_keyword.txt")
    ap.add_argument("--stream-windows", type=int, default=50)
    ap.add_argument("--tts-words", type=int, default=50,
                    help="only used when --tts is passed")
    ap.add_argument("--tts", action="store_true",
                    help="synthesize TTS distractor words for the cohort "
                         "(off by default; ablation study found no F1 gain)")
    ap.add_argument("--no-tts", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    keyword = args.keyword
    if keyword is None:
        kw_file = os.path.join(PROJECT_ROOT, "selected_keyword.txt")
        keyword = open(kw_file).read().strip() if os.path.exists(kw_file) else ""

    t0 = time.perf_counter()
    ui.banner("AS-NORM IMPOSTOR COHORT", f"keyword: {keyword or '(none)'}")

    # Window length comes from the anchor so cohort windows match what the
    # detector will embed; fall back to 0.7s if no anchor exists yet.
    anchor_npz = anchor_path(keyword) if keyword else ""
    if keyword and os.path.exists(anchor_npz):
        window_samples = int(np.load(anchor_npz)["window_samples"])
    else:
        window_samples = int(0.7 * SAMPLE_RATE)
        ui.warn("no anchor found - using default 0.70s cohort window")
    ui.kv("stream windows", args.stream_windows)
    ui.kv("window length", f"{window_samples / SAMPLE_RATE:.2f}s")
    ui.kv("TTS distractors", f"{args.tts_words} words" if args.tts else "off")

    rng = np.random.default_rng(args.seed)
    # Same leakage guard as calibrate.py: skip chunks whose transcript holds
    # the keyword or a stem derivative. With no keyword or no transcripts.txt,
    # keyword_free_chunks returns every chunk, so this is a no-op there.
    neg_files = keyword_free_chunks(keyword) if keyword else None
    ui.step(f"sampling {args.stream_windows} random stream windows ...")
    cohort_audio = sample_stream_windows(window_samples, args.stream_windows,
                                         rng, files=neg_files)
    n_stream = len(cohort_audio)

    n_tts = 0
    if args.tts:
        from keyword_generator import load_tts, synthesize
        processor, tts_model, vocoder, xvectors = load_tts()
        words = [w for w in DISTRACTOR_WORDS if w != keyword][:args.tts_words]
        ui.step(f"synthesizing {len(words)} TTS distractor words ...")
        import torch
        for i, word in enumerate(words):
            idx = int(rng.integers(0, len(xvectors)))
            xvec = torch.tensor(xvectors[idx]["xvector"])
            audio = synthesize(processor, tts_model, vocoder, word, xvec)
            trimmed, _ = librosa.effects.trim(audio, top_db=30)
            if len(trimmed) > 0.15 * SAMPLE_RATE:
                cohort_audio.append(trimmed.astype(np.float32))
                n_tts += 1
            if (i + 1) % 10 == 0:
                ui.item(f"{i + 1}/{len(words)} distractors done")

    model = load_siamese_model()
    ui.step(f"embedding {len(cohort_audio)} cohort clips ...")
    embeddings = embed_batch(model, cohort_audio)

    out_path = cohort_path(keyword)
    np.savez(out_path, embeddings=embeddings.astype(np.float32),
             n_stream=np.int64(n_stream), n_tts=np.int64(n_tts))
    ui.ok(f"cohort: {n_stream} stream windows + {n_tts} TTS distractors")
    ui.done(os.path.relpath(out_path, PROJECT_ROOT), t0)
    ui.item("next: python pipeline/calibrate.py")


if __name__ == "__main__":
    main()
