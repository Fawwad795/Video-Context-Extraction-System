"""Dense continuous-speech hard-negative training + evaluation (roadmap phase P4.5).

The P4 live test showed the matcher over-fires on real broadcast audio. Local analysis refined
the cause: the model discriminates the *topic* keyword well (journalism 0.888 vs unrelated
words ~0.65 on the live chunks), but the FLOOR for arbitrary DENSE speech is too high. Crucially,
concatenated Speech Commands clips are silence-padded and too easy (the P2 model already rejects
them at FPR ~0.007) -- they are NOT representative of dense broadcast speech.

So the hard negatives here are random 1-second windows of DENSE continuous speech (LibriSpeech),
which is the right domain. Positives keep clean keywords plus keyword-in-dense-context examples
so recall stays high while the dense-speech floor is pulled down.
"""
from __future__ import annotations

import glob
import os
import random
from collections import defaultdict

import numpy as np
import soundfile as sf
import torch

from .audio import fix_length, log_mel
from .config import AUDIO, NOISE_LABEL


def time_shift(wav: torch.Tensor, rng: random.Random, max_frac: float = 0.25) -> torch.Tensor:
    """Random pad+trim time-shift to move the word off-centre while keeping it fully present."""
    n = AUDIO.num_samples
    wav = fix_length(wav)
    sh = rng.randint(-int(max_frac * n), int(max_frac * n))
    if sh > 0:
        wav = torch.cat([torch.zeros(sh), wav])[:n]
    elif sh < 0:
        wav = torch.cat([wav, torch.zeros(-sh)])[-n:]
    return wav


class DenseSpeechWindows:
    """Random 1-second windows from a dense-speech corpus (LibriSpeech .flac files).

    Reads only the requested window via soundfile seek, so it never loads whole files.
    """

    def __init__(self, root: str):
        self.files = glob.glob(os.path.join(root, "**", "*.flac"), recursive=True)
        if not self.files:
            raise FileNotFoundError(f"no .flac under {root} (download LibriSpeech first)")

    def random_window(self, rng: random.Random) -> torch.Tensor:
        n = AUDIO.num_samples
        for _ in range(8):
            f = rng.choice(self.files)
            try:
                frames = sf.info(f).frames
            except Exception:
                continue
            if frames < n:
                continue
            start = rng.randint(0, frames - n)
            data, sr = sf.read(f, frames=n, start=start, dtype="float32", always_2d=False)
            wav = torch.from_numpy(data)
            if wav.dim() > 1:
                wav = wav.mean(dim=1)
            return fix_length(wav)
        return torch.zeros(n)


def _group_by_label(base) -> dict[str, list[int]]:
    by = defaultdict(list)
    for i, lab in enumerate(base.labels):
        if lab != NOISE_LABEL:
            by[lab].append(i)
    return by


class ContinuousPairDataset(torch.utils.data.Dataset):
    """(reference, candidate) pairs that pull dense non-keyword speech away from word prototypes:
      - pos      (1): clean word vs time-shifted clean SAME word
      - pos_ctx  (1): clean word vs the same word overlaid on low-level dense speech
      - easy_neg (0): clean word vs a different clean word
      - hard_neg (0): clean word vs a random DENSE-speech window (LibriSpeech)  <- the fix
    """

    def __init__(self, base, dense: DenseSpeechWindows, n_pairs: int,
                 pos_frac=0.35, posctx_frac=0.15, easyneg_frac=0.15, seed=1):
        self.base = base
        self.dense = dense
        self.seed = seed
        rng = random.Random(seed)
        by = _group_by_label(base)
        labels = [l for l in by if len(by[l]) >= 2]
        plan = []
        for _ in range(n_pairs):
            r = rng.random()
            la = rng.choice(labels)
            a = rng.choice(by[la])
            if r < pos_frac:
                plan.append(("pos", a, rng.choice(by[la])))
            elif r < pos_frac + posctx_frac:
                plan.append(("pos_ctx", a, rng.choice(by[la])))
            elif r < pos_frac + posctx_frac + easyneg_frac:
                lb = rng.choice([l for l in labels if l != la])
                plan.append(("easy_neg", a, rng.choice(by[lb])))
            else:
                plan.append(("hard_neg", a, None))
        self.plan = plan

    def __len__(self):
        return len(self.plan)

    def __getitem__(self, i):
        kind, a, b = self.plan[i]
        rng = random.Random(self.seed * 1_000_003 + i)        # worker-safe, deterministic
        ref = log_mel(self.base.raw_waveform(a))
        if kind == "pos":
            cand = log_mel(time_shift(self.base.raw_waveform(b), rng))
            y = 1.0
        elif kind == "pos_ctx":
            word = fix_length(self.base.raw_waveform(b))
            bg = self.dense.random_window(rng)
            cand = log_mel(word + 0.15 * bg)                  # keyword dominant over quiet speech
            y = 1.0
        elif kind == "easy_neg":
            cand = log_mel(self.base.raw_waveform(b))
            y = 0.0
        else:  # hard_neg: dense continuous-speech window
            cand = log_mel(self.dense.random_window(rng))
            y = 0.0
        return ref, cand, torch.tensor(y)


@torch.no_grad()
def clean_prototype(model, base, keyword: str, k: int = 10, device="cpu") -> torch.Tensor:
    idxs = [i for i, l in enumerate(base.labels) if l == keyword][:k]
    specs = [log_mel(base.raw_waveform(i)) for i in idxs]
    embs = model.embed(torch.stack(specs).to(device)).cpu()
    return torch.nn.functional.normalize(embs.mean(0), dim=0)


@torch.no_grad()
def continuous_fpr_tpr(model, base, dense: DenseSpeechWindows, prototype, keyword, threshold,
                       n_neg=2000, n_pos=300, seed=2, device="cpu"):
    """FPR on dense-speech windows (LibriSpeech); TPR on (shifted) clean keyword windows."""
    rng = random.Random(seed)
    by = _group_by_label(base)
    neg_specs = [log_mel(dense.random_window(rng)) for _ in range(n_neg)]
    pos_specs = [log_mel(time_shift(base.raw_waveform(i), rng)) for i in by[keyword][:n_pos]]

    def scores(specs):
        out = []
        for k in range(0, len(specs), 256):
            e = model.embed(torch.stack(specs[k:k + 256]).to(device)).cpu()
            out.append((e @ prototype).numpy())
        return np.concatenate(out)

    neg_s, pos_s = scores(neg_specs), scores(pos_specs)
    return {
        "fpr": float((neg_s >= threshold).mean()),
        "tpr": float((pos_s >= threshold).mean()),
        "neg_mean": float(neg_s.mean()),
        "pos_mean": float(pos_s.mean()),
    }
