"""Google Speech Commands v2 -> same/different word pairs for Siamese training.

`SpeechCommandsWords` wraps torchaudio's SPEECHCOMMANDS using the dataset's *official*
train/val/test split lists, so our held-out test set is reproducible and comparable.
`PairDataset` turns a split into balanced (anchor, other, label) pairs where label=1 means
"same word" and label=0 means "different word".

Note: the 35 Speech Commands words are only the *training vocabulary*. Because the Siamese
network learns a general similarity function (not a fixed classifier), it generalizes to
arbitrary unseen keywords at inference time.
"""
import os
import random
from collections import defaultdict

import torch
import torchaudio
from torch.utils.data import Dataset

from .audio import load_wav, log_mel
from .config import AUDIO, NOISE_LABEL, PAIRS, SC_CORPUS, SC_ROOT

VALID_SUBSETS = ("training", "validation", "testing")


class SpeechCommandsWords(Dataset):
    """Yields (log_mel_spectrogram, label, speaker_id) for one official subset."""

    def __init__(self, subset: str):
        if subset not in VALID_SUBSETS:
            raise ValueError(f"subset must be one of {VALID_SUBSETS}, got {subset!r}")
        self.subset = subset
        self.ds = torchaudio.datasets.SPEECHCOMMANDS(
            root=str(SC_ROOT), download=False, subset=subset
        )
        # Base dir that get_metadata() relpaths are relative to (torchaudio's _archive).
        self._base = getattr(self.ds, "_archive", str(SC_CORPUS))
        # Cache metadata for every item (no audio decode -> fast).
        # SPEECHCOMMANDS.get_metadata(n) -> (relpath, sample_rate, label, speaker_id, utt_no)
        self._metas = [self.ds.get_metadata(i) for i in range(len(self.ds))]
        self.labels = [m[2] for m in self._metas]

    def __len__(self) -> int:
        return len(self.ds)

    def label_of(self, i: int) -> str:
        return self.labels[i]

    def __getitem__(self, i: int):
        relpath, sr, label, speaker, _ = self._metas[i]
        wav, sr = load_wav(os.path.join(self._base, relpath))
        if sr != AUDIO.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, AUDIO.sample_rate)
        return log_mel(wav), label, speaker


class PairDataset(Dataset):
    """Balanced same/different-word pairs built over a `SpeechCommandsWords` split.

    Pairs are sampled once at construction (a deterministic index list given the seed),
    so an epoch sees a fixed, reproducible set of pairs.
    """

    def __init__(
        self,
        base: SpeechCommandsWords,
        neg_per_pos: int | None = None,
        max_pairs: int | None = None,
        seed: int | None = None,
    ):
        self.base = base
        neg_per_pos = PAIRS.neg_per_pos if neg_per_pos is None else neg_per_pos
        max_pairs = PAIRS.max_pairs if max_pairs is None else max_pairs
        seed = PAIRS.seed if seed is None else seed
        rng = random.Random(seed)

        # Group sample indices by word label (excluding the reserved noise folder).
        by_label: dict[str, list[int]] = defaultdict(list)
        for idx, lab in enumerate(base.labels):
            if lab == NOISE_LABEL:
                continue
            by_label[lab].append(idx)
        labels = list(by_label)

        pairs: list[tuple[int, int, int]] = []
        for lab, idxs in by_label.items():
            if len(idxs) < 2:
                continue
            for a in idxs:
                # one positive (same-word) partner
                b = a
                while b == a:
                    b = rng.choice(idxs)
                pairs.append((a, b, 1))
                # neg_per_pos negatives (different word)
                for _ in range(neg_per_pos):
                    other = lab
                    while other == lab:
                        other = rng.choice(labels)
                    pairs.append((a, rng.choice(by_label[other]), 0))

        rng.shuffle(pairs)
        if max_pairs:
            pairs = pairs[:max_pairs]
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        a, b, y = self.pairs[i]
        spec_a, _, _ = self.base[a]
        spec_b, _, _ = self.base[b]
        return spec_a, spec_b, torch.tensor(y, dtype=torch.float32)
