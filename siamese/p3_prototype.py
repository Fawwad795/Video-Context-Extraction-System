"""P3 (part B) - synthesized-keyword (prototype) vs real-reference matching.

This tests the *deployment* matching mode. In the live VMS the keyword reference is a
SYNTHETIC voice (SpeechT5 + HiFiGAN, 7 accents) averaged into one "prototype" embedding,
matched against REAL broadcast speech. Here we:

  1. synthesize each word in 7 accents, embed, average -> per-word PROTOTYPE (synthetic ref);
  2. build a REAL reference per word (mean embedding of a few real training clips);
  3. score real TEST clips against all 35 references and report 35-way top-1 accuracy + AUC.

The prototype-vs-real gap quantifies how much the synthetic->real domain shift costs.

Requires TTS deps (transformers, datasets, sentencepiece). Heavy on first run (downloads
SpeechT5 and synthesizes 35x7 clips into siamese/data/tts_cache).

    python -m siamese.p3_prototype --ckpt siamese/checkpoints/siamese_full_v1.pt
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch
import torchaudio
from sklearn.metrics import roc_auc_score

from .audio import load_wav, log_mel
from .config import AUDIO
from .datasets import SpeechCommandsWords
from .model import build_encoder
from .tts_bridge import SPEECH_COMMANDS_WORDS, synthesize_word


@torch.no_grad()
def embed_wavs(model, wavs, device, batch=128):
    specs = [log_mel(w) for w in wavs]
    out = []
    for k in range(0, len(specs), batch):
        out.append(model.embed(torch.stack(specs[k:k + batch]).to(device)).cpu())
    return torch.cat(out)


def _clips_by_word(base, words, per_word, skip=0):
    by = defaultdict(list)
    for i, lab in enumerate(base.labels):
        if lab in words:
            by[lab].append(i)
    return {w: by[w][skip:skip + per_word] for w in words}


def build_synth_prototypes(model, words, device):
    """Average the 7 synthetic-accent embeddings per word into one unit prototype vector."""
    protos = {}
    for n, w in enumerate(words, 1):
        wavs = []
        for p in synthesize_word(w):                       # cached after first run
            x, sr = load_wav(p)
            if sr != AUDIO.sample_rate:
                x = torchaudio.functional.resample(x, sr, AUDIO.sample_rate)
            wavs.append(x)
        e = embed_wavs(model, wavs, device)
        protos[w] = torch.nn.functional.normalize(e.mean(0), dim=0)
        print(f"  [{n}/{len(words)}] prototype '{w}'")
    return protos


def build_real_refs(model, train, words, k, device):
    """Reference = mean embedding of k real TRAIN clips per word (disjoint from the test set)."""
    clips = _clips_by_word(train, set(words), k)
    refs = {}
    for w in words:
        wavs = [train.raw_waveform(i) for i in clips[w]]
        e = embed_wavs(model, wavs, device)
        refs[w] = torch.nn.functional.normalize(e.mean(0), dim=0)
    return refs


def evaluate_refs(model, refs, test, words, per_word, device):
    """Score real TEST clips against every reference; return 35-way top-1 acc + AUC."""
    word_list = list(words)
    R = torch.stack([refs[w] for w in word_list])           # (W, d)
    clips = _clips_by_word(test, set(words), per_word)
    wavs, true_idx = [], []
    for w in word_list:
        for i in clips[w]:
            wavs.append(test.raw_waveform(i))
            true_idx.append(word_list.index(w))
    E = embed_wavs(model, wavs, device)                     # (N, d)
    S = (E @ R.t()).numpy()                                  # (N, W) cosine
    true = np.array(true_idx)
    top1 = float((S.argmax(1) == true).mean())
    onehot = np.zeros_like(S, dtype=int)
    onehot[np.arange(len(true)), true] = 1
    auc = roc_auc_score(onehot.ravel(), S.ravel())
    return top1, auc, len(true)


def main():
    ap = argparse.ArgumentParser(description="P3 synthetic-prototype vs real-reference eval.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--per-word", type=int, default=30, help="real test clips per word")
    ap.add_argument("--real-ref-k", type=int, default=10, help="real train clips per real ref")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = build_encoder(ckpt["model"]).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    words = SPEECH_COMMANDS_WORDS
    print(f"checkpoint: {args.ckpt} (model={ckpt['model']}) | {len(words)} words")

    print("\nSynthesizing + embedding keyword prototypes (7 accents each)...")
    synth = build_synth_prototypes(model, words, device)
    print("Building real references from training clips...")
    real = build_real_refs(model, SpeechCommandsWords("training"), words, args.real_ref_k, device)

    test = SpeechCommandsWords("testing")
    s_top1, s_auc, n = evaluate_refs(model, synth, test, words, args.per_word, device)
    r_top1, r_auc, _ = evaluate_refs(model, real, test, words, args.per_word, device)

    print(f"\n==== P3-B: matching {n} real test clips against 35 references ====")
    print(f"  {'reference':<22}{'top-1 acc':>10}{'AUC':>8}")
    print(f"  {'synthetic prototype':<22}{s_top1:>10.3f}{s_auc:>8.3f}")
    print(f"  {'real (train clips)':<22}{r_top1:>10.3f}{r_auc:>8.3f}")
    print(f"  synthetic->real gap   {r_top1 - s_top1:>10.3f}{r_auc - s_auc:>8.3f}")
    print("\n  (chance top-1 for 35 words = 0.029)")
    print("P3 part B done.")


if __name__ == "__main__":
    main()
