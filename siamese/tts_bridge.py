"""Synthetic <-> real bridge.

Synthesize keywords with the VMS's own TTS stack (SpeechT5 + HiFi-GAN, mirroring
Research/Stream1_hifigan.py) and pair the synthetic clips against *real* Speech Commands
utterances of the same word as POSITIVE pairs. This teaches the Siamese network to match a
synthetic reference to a real human voice — the exact domain gap the live system faces, and
the basis for the multi-accent "prototype" reference used at inference.

Heavy: downloads SpeechT5 (~600 MB) and runs synthesis. Intended to be run *once* to
populate a cache (ideally on the GPU box), then reused. Lazy imports keep the rest of the
P1 pipeline importable without transformers installed.

CLI:
    python -m siamese.tts_bridge --words yes no stop go        # cache specific words
    python -m siamese.tts_bridge --all-speech-commands         # cache all 35 SC words
"""
import argparse
import os

import soundfile as sf
import torch

from .config import TTS_CACHE_DIR

# 7 accents, identical to Research/Stream1_hifigan.py
SPEAKERS = {
    "awb": 0,     # Scottish male
    "bdl": 1138,  # US male
    "clb": 2271,  # US female
    "jmk": 3403,  # Canadian male
    "ksp": 4535,  # Indian male
    "rms": 5667,  # US male
    "slt": 6799,  # US female
}

# The 35 words in Speech Commands v0.02.
SPEECH_COMMANDS_WORDS = [
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow",
    "forward", "four", "go", "happy", "house", "learn", "left", "marvin", "nine",
    "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three",
    "tree", "two", "up", "visual", "wow", "yes", "zero",
]

_TTS = None  # lazily-loaded (processor, model, vocoder, speaker_embeddings)


def _load_tts():
    """Load SpeechT5 processor/model/vocoder + speaker embeddings on first use."""
    global _TTS
    if _TTS is not None:
        return _TTS
    from datasets import load_dataset
    from transformers import (SpeechT5ForTextToSpeech, SpeechT5HifiGan,
                              SpeechT5Processor)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
    # datasets>=4 dropped loading-script support; this dataset uses one, so request it
    # explicitly. Requires datasets<4 installed.
    embeddings = load_dataset(
        "Matthijs/cmu-arctic-xvectors", split="validation", trust_remote_code=True
    )
    _TTS = (processor, model, vocoder, embeddings, device)
    return _TTS


def synthesize_word(word: str, overwrite: bool = False) -> list[str]:
    """Synthesize `word` in all 7 accents, caching each as a 16 kHz WAV. Returns file paths."""
    TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    pending = {
        spk: TTS_CACHE_DIR / f"{spk}-{word}.wav"
        for spk in SPEAKERS
    }
    if not overwrite and all(p.exists() for p in pending.values()):
        return [str(p) for p in pending.values()]

    processor, model, vocoder, embeddings, device = _load_tts()
    inputs = processor(text=word, return_tensors="pt").to(device)
    for spk, sid in SPEAKERS.items():
        out_path = pending[spk]
        if out_path.exists() and not overwrite:
            paths.append(str(out_path))
            continue
        spk_emb = torch.tensor(embeddings[sid]["xvector"]).unsqueeze(0).to(device)
        with torch.no_grad():
            speech = model.generate_speech(inputs["input_ids"], spk_emb, vocoder=vocoder)
        sf.write(str(out_path), speech.cpu().numpy(), samplerate=16_000)
        paths.append(str(out_path))
    return paths


def precompute(words: list[str], overwrite: bool = False) -> None:
    for i, w in enumerate(words, 1):
        synthesize_word(w, overwrite=overwrite)
        print(f"[{i}/{len(words)}] cached '{w}' x{len(SPEAKERS)} accents")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Cache synthesized keyword references.")
    ap.add_argument("--words", nargs="*", default=[], help="words to synthesize")
    ap.add_argument("--all-speech-commands", action="store_true",
                    help="synthesize all 35 Speech Commands words")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    targets = list(args.words)
    if args.all_speech_commands:
        targets = SPEECH_COMMANDS_WORDS
    if not targets:
        ap.error("provide --words ... or --all-speech-commands")
    precompute(targets, overwrite=args.overwrite)
    print(f"Done. Cache: {TTS_CACHE_DIR}")
