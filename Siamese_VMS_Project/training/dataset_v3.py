"""Phase-3 dataset: cached frozen-backbone frames for word classification.

Step 3 of the siamese/optimizations plan trains ONLY a small attentive
pooling head (core/embedders.AttentivePoolingHead) on top of frozen WavLM
mid-layer frame features. The backbone never changes, so every clip's
frame features are extracted once into an on-disk cache and training
epochs never touch the backbone - a head epoch over ~20k clips takes
seconds, even on CPU.

Data sources (same corpora as Phase 2, different framing):
  real  - MSWC (MLCommons/ml_spoken_words) clips of the top-N word
          classes; broadcast-adjacent read speech, thousands of speakers.
  tts   - the Phase-2 TTS bank (training/tts_bank.py): multi-voice
          synthetic clips of the same words, half of them "humanized"
          (core/augment_utils). Sharing a CLASS between real and TTS
          clips makes the classification loss pull the two domains
          together per word - the job the Phase-2 GRL was doing, without
          the adversarial instability.

Hard negatives: sub-center ArcFace only sees other classes through the
batch's softmax denominator... which is fine for the loss itself, but
batches of random words are dominated by easy negatives. The
ConfusableBatchSampler builds each batch as P classes x K samples where
roughly half the classes are phonetic near-neighbors of a seed word
(edit distance over G2P phoneme strings when g2p_en is available,
grapheme strings otherwise - the GraphemeAug/PLCL confusable idea), so
every batch contains the minimal pairs the detector actually confuses.

Eval words from the TTS-bank manifest are excluded from training classes;
train_siamese_v3.py scores them zero-shot (TTS centroid vs real clips).

Build the cache once (GPU box):
    python dataset_v3.py --build-cache --manifest ~/tts_bank/manifest.json
"""

import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "core"))

SR = 16000
MAX_CLIP_SECONDS = 2.0
DEFAULT_CACHE = os.path.expanduser("~/v3_feature_cache")


# ------------------------------------------------------------ phonetic distance

def _levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def phonetic_neighbors(words, top_n=10):
    """word -> list of nearest confusable words (normalized edit distance).

    Uses g2p_en phoneme sequences when installed (ARPAbet, stress digits
    stripped), else falls back to grapheme sequences - a decent proxy for
    English minimal pairs and dependency-free.
    """
    try:
        from g2p_en import G2p
        g2p = G2p()
        seqs = {w: tuple(p.rstrip("012") for p in g2p(w) if p.strip() and p != " ")
                for w in words}
        print(f"Confusables from g2p_en phonemes ({len(words)} words).")
    except Exception:
        seqs = {w: tuple(w) for w in words}
        print(f"g2p_en unavailable - grapheme-distance confusables ({len(words)} words).")

    neighbors = {}
    wl = list(words)
    for w in wl:
        sw = seqs[w]
        scored = []
        for v in wl:
            if v == w:
                continue
            sv = seqs[v]
            if abs(len(sv) - len(sw)) > max(2, len(sw) // 2):
                continue
            d = _levenshtein(sw, sv) / max(len(sw), len(sv))
            scored.append((d, v))
        scored.sort()
        neighbors[w] = [v for _, v in scored[:top_n]]
    return neighbors


# ------------------------------------------------------------------ cache build

def select_classes(dataset, eval_words, n_classes, min_samples):
    from collections import Counter
    counts = Counter(k for k in dataset["keyword"] if k)
    eval_set = set(eval_words)
    picked = [w for w, c in counts.most_common()
              if c >= min_samples and w.isascii() and w.isalpha()
              and len(w) >= 3 and w not in eval_set]
    return picked[:n_classes]


def build_cache(args):
    """Extract WavLM mid-layer frames for every training/eval clip once.

    Layout: <cache>/<word>.npz  with arrays f0..fN (fp16 [T, D]) and a
    parallel JSON meta {word: [{"key", "domain"}...]}, plus eval_/ real
    clips for the held-out words.
    """
    import librosa
    from augment_utils import augment_audio
    from datasets import load_dataset
    from embedders import FrameBackend

    os.makedirs(args.cache_dir, exist_ok=True)
    with open(os.path.expanduser(args.manifest)) as f:
        manifest = json.load(f)

    print("Loading MSWC...")
    ds = load_dataset("MLCommons/ml_spoken_words", "en_wav", split="train",
                      trust_remote_code=True)
    class_to_indices = defaultdict(list)
    for idx, label in enumerate(ds["keyword"]):
        if label:
            class_to_indices[label].append(idx)

    eval_words = [w for w in manifest["eval_words"]
                  if w in class_to_indices and w in manifest["bank"]]
    classes = select_classes(ds, eval_words, args.n_classes, args.min_samples)
    print(f"{len(classes)} training classes, {len(eval_words)} eval words.")

    backend = FrameBackend(args.backbone)
    rng = np.random.default_rng(args.seed)
    max_samples = int(MAX_CLIP_SECONDS * SR)

    def clip_frames(audio):
        audio = np.asarray(audio, dtype=np.float32)[:max_samples]
        if len(audio) < int(0.15 * SR):
            return None
        return backend.frames(audio, [args.layer])[args.layer].astype(np.float16)

    def mswc_audio(idx):
        item = ds[int(idx)]
        audio = item["audio"]["array"]
        if item["audio"]["sampling_rate"] != SR:
            audio = librosa.resample(np.asarray(audio, dtype=np.float32),
                                     orig_sr=item["audio"]["sampling_rate"],
                                     target_sr=SR)
        return audio

    meta = {"backbone": args.backbone, "layer": args.layer,
            "classes": {}, "eval": {}}
    meta_path = os.path.join(args.cache_dir, "meta.json")

    for wi, word in enumerate(classes):
        out_path = os.path.join(args.cache_dir, f"{word}.npz")
        if os.path.exists(out_path):
            with np.load(out_path, allow_pickle=False) as z:
                meta["classes"][word] = json.loads(str(z["meta"]))
            continue
        feats, rows = [], []
        idxs = class_to_indices[word]
        rng.shuffle(idxs)
        for idx in idxs[:args.real_per_class]:
            fr = clip_frames(mswc_audio(idx))
            if fr is not None:
                feats.append(fr)
                rows.append({"domain": 0})
        for path in manifest["bank"].get(word, [])[:args.tts_per_class]:
            audio, _ = librosa.load(path, sr=SR)
            for aug in (False, True):
                a = augment_audio(audio, rng, sr=SR) if aug else audio
                fr = clip_frames(a)
                if fr is not None:
                    feats.append(fr)
                    rows.append({"domain": 1})
        if len(feats) < args.min_samples:
            continue
        np.savez(out_path, meta=json.dumps(rows),
                 **{f"f{i}": f for i, f in enumerate(feats)})
        meta["classes"][word] = rows
        if (wi + 1) % 25 == 0:
            print(f"[{wi + 1}/{len(classes)}] classes cached")
            with open(meta_path, "w") as f:
                json.dump(meta, f)

    # Held-out eval words: TTS clips (anchor side) + real clips (test side)
    for word in eval_words:
        out_path = os.path.join(args.cache_dir, f"eval_{word}.npz")
        if os.path.exists(out_path):
            with np.load(out_path, allow_pickle=False) as z:
                meta["eval"][word] = json.loads(str(z["meta"]))
            continue
        feats, rows = [], []
        for path in manifest["bank"][word][:3]:
            audio, _ = librosa.load(path, sr=SR)
            fr = clip_frames(audio)
            if fr is not None:
                feats.append(fr)
                rows.append({"domain": 1})
        idxs = list(class_to_indices[word])
        rng.shuffle(idxs)
        for idx in idxs[:args.eval_real_per_word]:
            fr = clip_frames(mswc_audio(idx))
            if fr is not None:
                feats.append(fr)
                rows.append({"domain": 0})
        if sum(r["domain"] == 1 for r in rows) and sum(r["domain"] == 0 for r in rows):
            np.savez(out_path, meta=json.dumps(rows),
                     **{f"f{i}": f for i, f in enumerate(feats)})
            meta["eval"][word] = rows

    with open(meta_path, "w") as f:
        json.dump(meta, f)
    n_train = sum(len(v) for v in meta["classes"].values())
    print(f"Cache complete: {len(meta['classes'])} classes / {n_train} clips, "
          f"{len(meta['eval'])} eval words -> {args.cache_dir}")


# --------------------------------------------------------------------- dataset

class CachedWordDataset(Dataset):
    """(frames fp32 [T, D], class_id, domain) from the feature cache."""

    def __init__(self, cache_dir=DEFAULT_CACHE):
        with open(os.path.join(cache_dir, "meta.json")) as f:
            self.meta = json.load(f)
        self.cache_dir = cache_dir
        self.words = sorted(self.meta["classes"])
        self.word_to_id = {w: i for i, w in enumerate(self.words)}
        self.items = []            # (word, clip_index, domain)
        self.class_items = defaultdict(list)
        for w in self.words:
            for ci, row in enumerate(self.meta["classes"][w]):
                self.class_items[w].append(len(self.items))
                self.items.append((w, ci, row["domain"]))
        print(f"CachedWordDataset: {len(self.words)} classes, "
              f"{len(self.items)} clips (backbone {self.meta['backbone']} "
              f"L{self.meta['layer']}).")

    def features(self, word, clip_index, prefix=""):
        # Open per access, never cache NpzFile handles: forked DataLoader
        # workers would share the parent's file descriptors and race on the
        # seek offset (manifested as spurious BadZipFile CRC errors).
        path = os.path.join(self.cache_dir, f"{prefix}{word}.npz")
        with np.load(path, allow_pickle=False) as z:
            return z[f"f{clip_index}"].astype(np.float32)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        word, ci, domain = self.items[i]
        fr = self.features(word, ci)
        # temporal jitter: random 85-100% crop, cheap stand-in for
        # signal-level tempo/offset augmentation (features are frozen)
        T = fr.shape[0]
        if T > 12 and random.random() < 0.5:
            keep = max(10, int(T * random.uniform(0.85, 1.0)))
            start = random.randint(0, T - keep)
            fr = fr[start:start + keep]
        return torch.from_numpy(fr), self.word_to_id[word], domain

    def eval_pack(self):
        """{word: {"tts": [frames...], "real": [frames...]}} for eval words."""
        pack = {}
        for w, rows in self.meta["eval"].items():
            tts = [self.features(w, i, "eval_") for i, r in enumerate(rows)
                   if r["domain"] == 1]
            real = [self.features(w, i, "eval_") for i, r in enumerate(rows)
                    if r["domain"] == 0]
            pack[w] = {"tts": tts, "real": real}
        return pack


def collate_pad(batch):
    """Pad to the batch max length; returns (frames, mask, labels, domains)."""
    T = max(f.shape[0] for f, _, _ in batch)
    D = batch[0][0].shape[1]
    frames = torch.zeros(len(batch), T, D)
    mask = torch.zeros(len(batch), T, dtype=torch.bool)
    labels = torch.tensor([l for _, l, _ in batch], dtype=torch.long)
    domains = torch.tensor([d for _, _, d in batch], dtype=torch.float32)
    for i, (f, _, _) in enumerate(batch):
        frames[i, :f.shape[0]] = f
        mask[i, :f.shape[0]] = True
    return frames, mask, labels, domains


class ConfusableBatchSampler(Sampler):
    """P classes x K samples per batch; ~half the classes are phonetic
    near-neighbors of a random seed word, so hard negatives share every
    batch with their confusable positives."""

    def __init__(self, dataset, classes_per_batch=16, samples_per_class=4,
                 batches_per_epoch=250, p_confusable=0.5, neighbor_top_n=10,
                 seed=0):
        self.ds = dataset
        self.P, self.K = classes_per_batch, samples_per_class
        self.n_batches = batches_per_epoch
        self.p_confusable = p_confusable
        self.rng = random.Random(seed)
        self.neighbors = phonetic_neighbors(self.ds.words, neighbor_top_n)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            seed_word = self.rng.choice(self.ds.words)
            group = [seed_word]
            n_conf = int(self.P * self.p_confusable)
            group += self.neighbors[seed_word][:n_conf - 1]
            pool = [w for w in self.ds.words if w not in set(group)]
            group += self.rng.sample(pool, self.P - len(group))
            batch = []
            for w in group:
                items = self.ds.class_items[w]
                take = (self.rng.sample(items, self.K) if len(items) >= self.K
                        else self.rng.choices(items, k=self.K))
                batch.extend(take)
            self.rng.shuffle(batch)
            yield batch


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the Phase-3 feature cache.")
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--manifest", default=os.path.expanduser("~/tts_bank/manifest.json"))
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    ap.add_argument("--backbone", default="microsoft/wavlm-base-plus")
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--n-classes", type=int, default=1000)
    ap.add_argument("--min-samples", type=int, default=8)
    ap.add_argument("--real-per-class", type=int, default=25)
    ap.add_argument("--tts-per-class", type=int, default=5)
    ap.add_argument("--eval-real-per-word", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.build_cache:
        build_cache(args)
    else:
        ap.print_help()
