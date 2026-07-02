"""Phase-2 triplet dataset: real MSWC speech mixed with TTS-bank clips.

Two changes vs. dataset.py:

1. Cross-domain positive pairs - with probability p_cross_domain, the
   (anchor, positive) pair for a word is one real human clip and one TTS
   clip of the same word, so "same word, different domain" is explicitly
   pulled together by the triplet loss. TTS clips are randomly "humanized"
   (augment_utils) for diversity.
2. Domain labels - every clip carries real=0 / synthetic=1, consumed by the
   gradient-reversal domain classifier in train_siamese_v2.py.

Eval words listed in the TTS-bank manifest are fully excluded from training
triplets so the cross-domain validation metric stays zero-shot.
"""

import json
import os
import random
from collections import defaultdict

import librosa
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

# Shared modules (scoring, siamese_model, augment_utils) live in ../core
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "core"))

from augment_utils import augment_audio

SR = 16000
CLIP_SAMPLES = 16000  # 1 second, same as v1 training


class SpeechTripletDomainDataset(Dataset):
    def __init__(self, manifest_path="~/tts_bank/manifest.json", num_samples=10000,
                 p_cross_domain=0.5, p_tts_negative=0.3, p_augment_tts=0.7):
        manifest_path = os.path.expanduser(manifest_path)
        print("Loading MLCommons/ml_spoken_words dataset...")
        self.dataset = load_dataset("MLCommons/ml_spoken_words", "en_wav",
                                    split="train", trust_remote_code=True)

        print("Grouping dataset indices by keyword class...")
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(self.dataset["keyword"]):
            if label:
                self.class_to_indices[label].append(idx)

        with open(manifest_path) as f:
            manifest = json.load(f)
        self.eval_words = [w for w in manifest["eval_words"]
                           if w in self.class_to_indices]
        # Snapshot real-sample indices for eval words BEFORE excluding them
        self.eval_class_to_indices = {w: list(self.class_to_indices[w])
                                      for w in self.eval_words}
        self.eval_bank = {w: manifest["bank"][w] for w in self.eval_words
                          if w in manifest["bank"]}

        eval_set = set(self.eval_words)
        self.classes = [c for c, idxs in self.class_to_indices.items()
                        if len(idxs) >= 3 and c not in eval_set]
        self.bank = {w: paths for w, paths in manifest["bank"].items()
                     if w in self.class_to_indices and w not in eval_set}

        self.num_samples = num_samples
        self.p_cross_domain = p_cross_domain
        self.p_tts_negative = p_tts_negative
        self.p_augment_tts = p_augment_tts
        print(f"Dataset ready: {len(self.classes)} training classes "
              f"({len(self.bank)} with TTS clips), "
              f"{len(self.eval_words)} eval words held out.")

    def __len__(self):
        return self.num_samples

    @staticmethod
    def fix_length(audio):
        if len(audio) > CLIP_SAMPLES:
            return audio[:CLIP_SAMPLES].astype(np.float32)
        return np.pad(audio, (0, CLIP_SAMPLES - len(audio))).astype(np.float32)

    def get_real_clip(self, idx):
        item = self.dataset[idx]
        audio = item["audio"]["array"]
        sr = item["audio"]["sampling_rate"]
        if sr != SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
        return self.fix_length(audio)

    def get_tts_clip(self, word, rng, augment=True):
        paths = self.bank.get(word) or self.eval_bank.get(word)
        path = paths[int(rng.integers(len(paths)))]
        audio, _ = librosa.load(path, sr=SR)
        if augment and rng.random() < self.p_augment_tts:
            audio = augment_audio(audio, rng, sr=SR)
        return self.fix_length(audio)

    def __getitem__(self, _):
        rng = np.random.default_rng(random.getrandbits(32))

        anchor_class = random.choice(self.classes)
        negative_class = random.choice(self.classes)
        while negative_class == anchor_class:
            negative_class = random.choice(self.classes)

        cross = anchor_class in self.bank and random.random() < self.p_cross_domain
        if cross:
            real = self.get_real_clip(random.choice(self.class_to_indices[anchor_class]))
            tts = self.get_tts_clip(anchor_class, rng)
            if random.random() < 0.5:
                anchor, d_a, positive, d_p = tts, 1.0, real, 0.0
            else:
                anchor, d_a, positive, d_p = real, 0.0, tts, 1.0
        else:
            a_idx, p_idx = random.sample(self.class_to_indices[anchor_class], 2)
            anchor, positive = self.get_real_clip(a_idx), self.get_real_clip(p_idx)
            d_a = d_p = 0.0

        if negative_class in self.bank and random.random() < self.p_tts_negative:
            negative, d_n = self.get_tts_clip(negative_class, rng), 1.0
        else:
            negative = self.get_real_clip(random.choice(self.class_to_indices[negative_class]))
            d_n = 0.0

        return (anchor, positive, negative,
                np.float32(d_a), np.float32(d_p), np.float32(d_n))


def _worker_init(worker_id):
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def make_dataloader(batch_size=32, num_workers=4, **dataset_kwargs):
    ds = SpeechTripletDomainDataset(**dataset_kwargs)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, worker_init_fn=_worker_init)
    return ds, loader
