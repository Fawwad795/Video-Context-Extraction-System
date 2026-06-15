"""Generate a multi-voice TTS word bank on the GPU instance (Phase 2).

The bank supplies synthetic positives for domain-mixed triplet training:
the most frequent MSWC keyword classes are synthesized with several voices
each (canonical CMU ARCTIC speakers, random utterance x-vectors, and
cross-speaker blends). A held-out set of eval words also gets TTS clips -
they are used ONLY for the cross-domain validation metric (TTS anchor vs
real speech) and are excluded from training triplets by dataset_v2.py.

Resume-safe: existing wavs are kept, the manifest is flushed periodically.

Output: <out>/<word>/<i>.wav and <out>/manifest.json
"""

import argparse
import glob
import json
import os
from collections import Counter

import librosa
import numpy as np
import soundfile as sf
import torch

SR = 16000
CANONICAL = [0, 1138, 2271, 3403, 4535, 5667, 6799]  # 7 CMU ARCTIC speakers


def pick_xvector(xvectors, rng):
    r = rng.random()
    if r < 0.3:
        idx = CANONICAL[int(rng.integers(len(CANONICAL)))]
        return torch.tensor(xvectors[idx]["xvector"])
    if r < 0.7:
        idx = int(rng.integers(len(xvectors)))
        return torch.tensor(xvectors[idx]["xvector"])
    i1 = int(rng.integers(len(xvectors)))
    i2 = int(rng.integers(len(xvectors)))
    alpha = float(rng.uniform(0.3, 0.7))
    blend = alpha * np.array(xvectors[i1]["xvector"]) \
        + (1.0 - alpha) * np.array(xvectors[i2]["xvector"])
    return torch.tensor(blend, dtype=torch.float32)


def flush_manifest(out_dir, manifest):
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f)


def main():
    ap = argparse.ArgumentParser(description="Generate the Phase-2 TTS word bank.")
    ap.add_argument("--bank-words", type=int, default=1500)
    ap.add_argument("--eval-words", type=int, default=150)
    ap.add_argument("--voices", type=int, default=5)
    ap.add_argument("--min-class-samples", type=int, default=8)
    ap.add_argument("--out", default=os.path.expanduser("~/tts_bank"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    print("Loading MSWC keyword column (cached)...")
    from datasets import load_dataset
    ds = load_dataset("MLCommons/ml_spoken_words", "en_wav", split="train",
                      trust_remote_code=True)
    counts = Counter(k for k in ds["keyword"] if k)
    candidates = [w for w, c in counts.most_common()
                  if c >= args.min_class_samples and w.isascii()
                  and w.isalpha() and len(w) >= 3]
    need = args.bank_words + args.eval_words
    pool = candidates[:need]
    print(f"{len(candidates)} candidate classes; taking top {len(pool)}.")
    order = rng.permutation(len(pool))
    eval_words = [pool[i] for i in order[:args.eval_words]]
    bank_words = [pool[i] for i in order[args.eval_words:]]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SpeechT5 on {device}...")
    from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    # use_safetensors: transformers >= 4.56 blocks torch.load checkpoints on torch < 2.6
    tts = SpeechT5ForTextToSpeech.from_pretrained(
        "microsoft/speecht5_tts", use_safetensors=True).to(device)
    vocoder = SpeechT5HifiGan.from_pretrained(
        "microsoft/speecht5_hifigan", use_safetensors=True).to(device)
    xvectors = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation",
                            trust_remote_code=True)

    manifest = {"sample_rate": SR, "voices": args.voices,
                "eval_words": eval_words, "bank": {}}
    all_words = bank_words + eval_words
    n_clips = 0
    for wi, word in enumerate(all_words):
        word_dir = os.path.join(args.out, word)
        os.makedirs(word_dir, exist_ok=True)
        clips = sorted(glob.glob(os.path.join(word_dir, "*.wav")))
        if len(clips) < args.voices:
            for v in range(args.voices):
                path = os.path.join(word_dir, f"{v}.wav")
                if os.path.exists(path):
                    continue
                xvec = pick_xvector(xvectors, rng).to(device)
                inputs = processor(text=word, return_tensors="pt").to(device)
                with torch.no_grad():
                    speech = tts.generate_speech(
                        inputs["input_ids"], xvec.unsqueeze(0), vocoder=vocoder)
                audio = speech.cpu().numpy().astype(np.float32)
                trimmed, _ = librosa.effects.trim(audio, top_db=30)
                # degenerate-synthesis guards (babble / silence)
                if not (0.15 * SR <= len(trimmed) <= 2.5 * SR):
                    continue
                sf.write(path, trimmed, SR)
            clips = sorted(glob.glob(os.path.join(word_dir, "*.wav")))

        if clips:
            lens = {p: sf.info(p).frames for p in clips}
            med = float(np.median(list(lens.values())))
            keep = [p for p in clips if lens[p] <= 1.8 * med]
            for p in set(clips) - set(keep):
                os.remove(p)
            if keep:
                manifest["bank"][word] = keep
                n_clips += len(keep)

        if (wi + 1) % 50 == 0:
            print(f"[{wi + 1}/{len(all_words)}] words done, {n_clips} clips so far")
        if (wi + 1) % 100 == 0:
            flush_manifest(args.out, manifest)

    flush_manifest(args.out, manifest)
    n_eval = sum(1 for w in eval_words if w in manifest["bank"])
    print(f"TTS bank complete: {len(manifest['bank'])} words / {n_clips} clips "
          f"({n_eval} eval words held out). Manifest: {args.out}/manifest.json")


if __name__ == "__main__":
    main()
