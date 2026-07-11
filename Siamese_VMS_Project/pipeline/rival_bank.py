"""Build the global rival bank: frequent English words embedded once.

The bank is the candidate pool for embedding-space rival selection
(core/rival_verify.py). Words come from Whisper's BPE vocabulary in merge
rank order (an offline proxy for corpus frequency), filtered to alphabetic
words with CMUdict pronunciations; each word is synthesized in one canonical
voice and embedded with the active backend.

The bank is keyword-independent but embedding-dependent (named with the
backend suffix). It is a model-level artifact like a checkpoint, stored in
the code repo's keywords/ dir regardless of SIAMESE_PROJECT_ROOT (override
with SIAMESE_RIVAL_BANK). Building is resumable: progress is checkpointed
every --checkpoint words, and an interrupted or partial bank is usable
immediately (selection just ranks whatever is present).

Usage: python pipeline/rival_bank.py [--size 4000]
"""

import argparse
import os
import time
import warnings

import numpy as np

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "core"))

import console as ui
import rival_verify as rv
from scoring import SAMPLE_RATE, embed_batch, l2_normalize, load_siamese_model

warnings.filterwarnings("ignore")


def frequent_words(size):
    """Alphabetic words with CMUdict pronunciations, Whisper-BPE rank order."""
    from nltk.corpus import cmudict
    from transformers import WhisperTokenizer
    cd = cmudict.dict()
    tok = WhisperTokenizer.from_pretrained("openai/whisper-base")
    words, seen = [], set()
    for t, _ in sorted(tok.get_vocab().items(), key=lambda kv: kv[1]):
        if t.startswith("Ġ"):
            w = t[1:].lower()
            if w.isalpha() and len(w) >= 3 and w in cd and w not in seen:
                seen.add(w)
                words.append(w)
                if len(words) >= size:
                    break
    return words


def main():
    ap = argparse.ArgumentParser(description="Build the global rival bank.")
    ap.add_argument("--size", type=int, default=4000)
    ap.add_argument("--voice", default="bdl")
    ap.add_argument("--checkpoint", type=int, default=100)
    args = ap.parse_args()

    t0 = time.perf_counter()
    ui.banner("RIVAL BANK", f"{args.size} frequent words, voice '{args.voice}'")
    bank_path = rv.bank_path()

    done_words, done_embs = [], None
    if os.path.exists(bank_path):
        z = np.load(bank_path, allow_pickle=False)
        done_words = [str(w) for w in z["words"]]
        done_embs = z["embs"]
        ui.ok(f"resuming: {len(done_words)} words already embedded")

    words = [w for w in frequent_words(args.size) if w not in set(done_words)]
    if not words:
        ui.ok("bank complete")
        return
    ui.step(f"{len(words)} words to synthesize + embed ...")

    import librosa
    import torch
    from keyword_generator import CANONICAL_SPEAKERS, load_tts, synthesize
    processor, tts_model, vocoder, xvectors = load_tts()
    xv = torch.tensor(xvectors[CANONICAL_SPEAKERS[args.voice]]["xvector"])
    model = load_siamese_model()

    batch_words, batch_clips = [], []
    n_done = len(done_words)

    def flush():
        nonlocal done_words, done_embs, batch_words, batch_clips
        if not batch_clips:
            return
        embs = l2_normalize(embed_batch(model, batch_clips), axis=1)
        done_embs = embs if done_embs is None else np.vstack([done_embs, embs])
        done_words += batch_words
        os.makedirs(os.path.dirname(bank_path), exist_ok=True)
        np.savez(bank_path, words=np.array(done_words),
                 embs=done_embs.astype(np.float32))
        batch_words, batch_clips = [], []

    for i, w in enumerate(words):
        try:
            wav = synthesize(processor, tts_model, vocoder, w, xv)
            y, _ = librosa.effects.trim(wav, top_db=30)
            if len(y) < int(0.12 * SAMPLE_RATE):
                continue
            batch_words.append(w)
            batch_clips.append(y)
        except Exception as e:
            ui.warn(f"{w}: {type(e).__name__} - skipped")
        if len(batch_clips) >= args.checkpoint:
            flush()
            ui.item(f"{n_done + i + 1}/{n_done + len(words)} "
                    f"({(time.perf_counter() - t0) / 60:.1f} min)")
    flush()
    ui.done(f"{bank_path} ({len(done_words)} words)", t0)


if __name__ == "__main__":
    main()
