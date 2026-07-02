"""Build a multi-voice, augmented TTS *prototype anchor* for the keyword.

Phase-1 fix for the synthetic-to-real domain gap: a single TTS clip is a
brittle anchor because the embedding space strongly encodes "synthetic-ness"
and single-voice idiosyncrasies. Instead we:

  1. synthesize the keyword with many voices - the 7 canonical CMU ARCTIC
     speakers, random utterance-level x-vectors, and cross-speaker x-vector
     blends (novel voices via interpolation);
  2. "humanize" every clip with random augmentation - additive noise,
     synthetic room reverb, pitch shift, tempo stretch, band-limiting;
  3. embed all copies with the Siamese model, L2-normalize, and average the
     held-in voices into a single centroid anchor (prototype averaging).

A few voices are held out: their embeddings are saved as calibration
positives for calibrate.py.

Output: keywords/<keyword>_anchor.npz
        (centroid [D], positives [P, D], window_samples, keyword)
"""

import argparse
import os

import librosa
import numpy as np
import soundfile as sf
import torch

# Shared modules (scoring, siamese_model, augment_utils) live in ../core
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "core"))

from augment_utils import augment_audio
from scoring import PROJECT_ROOT, SAMPLE_RATE, embed_batch, l2_normalize, load_siamese_model

# Block-start indices of the 7 CMU ARCTIC speakers in Matthijs/cmu-arctic-xvectors
CANONICAL_SPEAKERS = {
    "awb": 0, "bdl": 1138, "clb": 2271, "jmk": 3403,
    "ksp": 4535, "rms": 5667, "slt": 6799,
}
MIN_CLIP_SECONDS = 0.15


def load_tts():
    from datasets import load_dataset
    from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

    print("Loading SpeechT5 models...")
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    tts_model = SpeechT5ForTextToSpeech.from_pretrained(
        "microsoft/speecht5_tts", use_safetensors=True)
    vocoder = SpeechT5HifiGan.from_pretrained(
        "microsoft/speecht5_hifigan", use_safetensors=True)
    print("Loading speaker x-vectors...")
    xvectors = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    return processor, tts_model, vocoder, xvectors


def synthesize(processor, tts_model, vocoder, text, xvector):
    inputs = processor(text=text, return_tensors="pt")
    with torch.no_grad():
        speech = tts_model.generate_speech(
            inputs["input_ids"], xvector.unsqueeze(0), vocoder=vocoder)
    return speech.numpy().astype(np.float32)


def pick_voices(xvectors, n_random, n_blend, rng):
    """Canonical speakers + random utterance x-vectors + cross-speaker blends."""
    voices = [(name, torch.tensor(xvectors[idx]["xvector"]))
              for name, idx in CANONICAL_SPEAKERS.items()]
    n_total = len(xvectors)
    for i in range(n_random):
        idx = int(rng.integers(0, n_total))
        voices.append((f"rand{i}_{idx}", torch.tensor(xvectors[idx]["xvector"])))
    for i in range(n_blend):
        i1 = int(rng.integers(0, n_total))
        i2 = int(rng.integers(0, n_total))
        alpha = float(rng.uniform(0.3, 0.7))
        blend = alpha * np.array(xvectors[i1]["xvector"]) \
            + (1.0 - alpha) * np.array(xvectors[i2]["xvector"])
        voices.append((f"blend{i}_{i1}x{i2}", torch.tensor(blend, dtype=torch.float32)))
    return voices


def main():
    ap = argparse.ArgumentParser(description="Build a multi-voice TTS prototype anchor.")
    ap.add_argument("--keyword", default=None, help="defaults to selected_keyword.txt")
    ap.add_argument("--n-random", type=int, default=13, help="random utterance x-vectors")
    ap.add_argument("--n-blend", type=int, default=10, help="cross-speaker x-vector blends")
    ap.add_argument("--n-augment", type=int, default=2, help="augmented copies per voice")
    ap.add_argument("--holdout", type=int, default=6, help="voices held out as calibration positives")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    keyword = args.keyword
    if keyword is None:
        kw_file = os.path.join(PROJECT_ROOT, "selected_keyword.txt")
        if not os.path.exists(kw_file):
            print("No --keyword given and selected_keyword.txt not found. Run transcriber.py first.")
            return
        with open(kw_file) as f:
            keyword = f.read().strip()
    print(f"Building prototype anchor for keyword: '{keyword}'")

    rng_voices = np.random.default_rng(args.seed)
    rng_aug = np.random.default_rng(args.seed + 1)
    variants_dir = os.path.join(PROJECT_ROOT, "keywords", f"{keyword}_variants")
    os.makedirs(variants_dir, exist_ok=True)

    # ---- 1. Synthesize the keyword with every voice (cached on disk) ----
    processor, tts_model, vocoder, xvectors = load_tts()
    voices = pick_voices(xvectors, args.n_random, args.n_blend, rng_voices)

    clips = []          # (voice_name, trimmed clean audio)
    for i, (name, xvec) in enumerate(voices):
        wav_path = os.path.join(variants_dir, f"{i:02d}_{name}.wav")
        if os.path.exists(wav_path):
            audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
        else:
            audio = synthesize(processor, tts_model, vocoder, keyword, xvec)
            sf.write(wav_path, audio, samplerate=SAMPLE_RATE)
        trimmed, _ = librosa.effects.trim(audio, top_db=30)
        if len(trimmed) < MIN_CLIP_SECONDS * SAMPLE_RATE:
            print(f"  Skipping voice {name}: synthesis too short after trim.")
            continue
        clips.append((name, trimmed.astype(np.float32)))
        print(f"  [{i + 1}/{len(voices)}] voice {name}: {len(trimmed) / SAMPLE_RATE:.2f}s")

    # Drop degenerate syntheses (SpeechT5 occasionally babbles/repeats for
    # unusual x-vectors, producing clips several times the median length)
    median_len = np.median([len(c) for _, c in clips])
    kept = []
    for name, c in clips:
        if len(c) > 1.8 * median_len:
            print(f"  Dropping voice {name}: {len(c) / SAMPLE_RATE:.2f}s "
                  f"(>1.8x median - degenerate synthesis)")
        else:
            kept.append((name, c))
    clips = kept

    if len(clips) <= args.holdout + 2:
        print("Not enough usable voices to build an anchor. Aborting.")
        return

    # The sliding-window length the detector will use
    window_samples = int(np.median([len(c) for _, c in clips]))

    # ---- 2. Augment + split voices into centroid set / held-out positives ----
    order = rng_aug.permutation(len(clips))
    holdout_idx = set(order[:args.holdout].tolist())

    centroid_audio, positive_audio = [], []
    for i, (name, clean) in enumerate(clips):
        copies = [clean] + [augment_audio(clean, rng_aug) for _ in range(args.n_augment)]
        (positive_audio if i in holdout_idx else centroid_audio).extend(copies)

    # ---- 3. Embed and average into the centroid anchor ----
    model = load_siamese_model()
    print(f"Embedding {len(centroid_audio)} centroid clips + {len(positive_audio)} held-out positives...")
    centroid_embs = embed_batch(model, centroid_audio)
    positives = embed_batch(model, positive_audio)

    # Two-pass centroid: drop embedding outliers (bad syntheses / augmentations
    # that landed far from the cloud), then re-average
    centroid = l2_normalize(centroid_embs.mean(axis=0))
    cos_to_centroid = centroid_embs @ centroid
    keep = cos_to_centroid >= 0.5
    if 0 < (~keep).sum() < 0.3 * len(centroid_embs):
        print(f"Trimming {(~keep).sum()} outlier embeddings (cos < 0.5) from the centroid.")
        centroid = l2_normalize(centroid_embs[keep].mean(axis=0))
        cos_to_centroid = centroid_embs[keep] @ centroid

    print(f"Centroid cohesion: cos(variant, centroid) "
          f"mean={cos_to_centroid.mean():.3f} min={cos_to_centroid.min():.3f}")

    out_path = os.path.join(PROJECT_ROOT, "keywords", f"{keyword}_anchor.npz")
    np.savez(out_path,
             centroid=centroid.astype(np.float32),
             positives=positives.astype(np.float32),
             window_samples=np.int64(window_samples),
             keyword=np.str_(keyword))
    print(f"Anchor saved: {out_path}")
    print(f"  voices used: {len(clips)} ({len(clips) - len(holdout_idx)} centroid, "
          f"{len(holdout_idx)} held out)  |  embeddings averaged: {len(centroid_audio)}")
    print(f"  detection window: {window_samples / SAMPLE_RATE:.2f}s")
    print("Next: python cohort_builder.py && python calibrate.py")


if __name__ == "__main__":
    main()
