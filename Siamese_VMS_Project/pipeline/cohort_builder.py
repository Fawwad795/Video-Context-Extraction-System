"""Build the AS-norm impostor cohort.

The cohort answers "how close does a *non*-keyword typically get to the
anchor / to a window?" - every detection score is normalized against it,
which is what replaces the hardcoded distance threshold.

Two impostor sources:
  1. random keyword-length windows sampled from the live chunks -
     domain-matched negatives (speech, music, silence of the actual stream);
  2. TTS distractor words in random voices - same-domain competition for the
     TTS anchor, so the anchor's synthetic-domain advantage is normalized
     away (skip with --no-tts).

Output: cohort.npz (embeddings [N, D])
"""

import argparse
import os

import librosa
import numpy as np

# Shared modules (scoring, siamese_model, augment_utils) live in ../core
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "core"))

from scoring import (PROJECT_ROOT, SAMPLE_RATE, cohort_path, embed_batch,
                     load_siamese_model, sample_stream_windows)

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
    ap.add_argument("--stream-windows", type=int, default=200)
    ap.add_argument("--tts-words", type=int, default=50)
    ap.add_argument("--no-tts", action="store_true", help="skip TTS distractor words")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    keyword = args.keyword
    if keyword is None:
        kw_file = os.path.join(PROJECT_ROOT, "selected_keyword.txt")
        keyword = open(kw_file).read().strip() if os.path.exists(kw_file) else ""

    # Window length comes from the anchor so cohort windows match what the
    # detector will embed; fall back to 0.7s if no anchor exists yet.
    anchor_path = os.path.join(PROJECT_ROOT, "keywords", f"{keyword}_anchor.npz")
    if keyword and os.path.exists(anchor_path):
        window_samples = int(np.load(anchor_path)["window_samples"])
    else:
        window_samples = int(0.7 * SAMPLE_RATE)
        print("No anchor found - using default 0.70s cohort window.")

    rng = np.random.default_rng(args.seed)
    print(f"Sampling {args.stream_windows} random stream windows "
          f"({window_samples / SAMPLE_RATE:.2f}s each)...")
    cohort_audio = sample_stream_windows(window_samples, args.stream_windows, rng)
    n_stream = len(cohort_audio)

    n_tts = 0
    if not args.no_tts:
        from keyword_generator import load_tts, synthesize
        processor, tts_model, vocoder, xvectors = load_tts()
        words = [w for w in DISTRACTOR_WORDS if w != keyword][:args.tts_words]
        print(f"Synthesizing {len(words)} TTS distractor words...")
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
                print(f"  {i + 1}/{len(words)} distractors done")

    model = load_siamese_model()
    print(f"Embedding {len(cohort_audio)} cohort clips...")
    embeddings = embed_batch(model, cohort_audio)

    out_path = cohort_path(keyword)
    np.savez(out_path, embeddings=embeddings.astype(np.float32),
             n_stream=np.int64(n_stream), n_tts=np.int64(n_tts))
    print(f"Cohort saved: {out_path} "
          f"({n_stream} stream windows + {n_tts} TTS distractors)")
    print("Next: python calibrate.py")


if __name__ == "__main__":
    main()
