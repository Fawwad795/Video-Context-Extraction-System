"""Phase-3 training: attentive pooling head + sub-center ArcFace (Step 3).

What changed vs. Phase 2 (train_siamese_v2.py):

  backbone   wav2vec2-base LAST layer -> frozen WavLM-base-plus mid layer
             (L10), where word identity actually lives (validated by
             pipeline/eval_scoring_ab.py: AP 0.586 -> 1.00 on Set D);
  pooling    frame mean -> trained multi-head attentive pooling
             (core/embedders.AttentivePoolingHead, identity-init so
             epoch 0 == the already-validated mean-pool operating point);
  loss       vanilla triplet + GRL -> sub-center ArcFace over ~1000 word
             classes. The angular margin directly optimizes what the
             detector thresholds (cosine to an anchor centroid): every
             real/TTS clip of a word must sit within a margin-tightened
             cone, which is exactly "lift REAL keyword scores above the
             calibrated threshold", not just ranking. Sub-centers (K=3)
             absorb the real-vs-TTS bimodality per class, replacing the
             Phase-2 GRL (which never converged with a frozen backbone).
  negatives  random triplets -> phonetic-confusable batches
             (dataset_v3.ConfusableBatchSampler).

The backbone is frozen and features are pre-cached (dataset_v3.py
--build-cache), so an epoch is head-only and takes seconds on GPU.

Validation: cross-domain AUC on held-out eval words - TTS-clip centroid
scored by cosine against real clips of the same word (positives) vs real
clips of other eval words (negatives). Zero-shot, same protocol as v2, so
numbers are comparable. Also reported: pos/neg cosine means, i.e. the
absolute-score lift Step 3 is about.

Checkpoints: siamese_v3_best.pth (best AUC), siamese_v3_final.pth -
consumed by SIAMESE_BACKEND=wavlm-trained (core/scoring.py).
"""

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "core"))

from dataset_v3 import (DEFAULT_CACHE, CachedWordDataset,
                        ConfusableBatchSampler, collate_pad)
from embedders import AttentivePoolingHead, save_head_checkpoint


class SubCenterArcFace(nn.Module):
    """Sub-center ArcFace (Deng et al., ECCV 2020) over word classes.

    K sub-centers per class; a sample's class logit is its best-matching
    sub-center, so intra-class modes (real vs TTS voices) need not
    collapse to one point. Margin m is added to the target angle,
    scale s sharpens the softmax.
    """

    def __init__(self, dim, n_classes, k=3, s=30.0, m=0.2):
        super().__init__()
        self.k, self.s, self.m = k, s, m
        self.weight = nn.Parameter(torch.empty(n_classes * k, dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, emb, labels):
        emb = F.normalize(emb, dim=-1)
        w = F.normalize(self.weight, dim=-1)
        cos = emb @ w.t()                                    # [B, C*K]
        cos = cos.view(len(emb), -1, self.k).max(dim=-1).values  # [B, C]
        theta = torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))
        target = F.one_hot(labels, cos.shape[1]).bool()
        logits = torch.where(target, torch.cos(theta + self.m), cos)
        return F.cross_entropy(self.s * logits, labels)


def embed_features(head, feats_list, device, batch_size=128):
    """List of [T, D] float32 arrays -> L2-normalized [N, D] (no grad)."""
    out = []
    with torch.no_grad():
        for i in range(0, len(feats_list), batch_size):
            chunk = feats_list[i:i + batch_size]
            T = max(f.shape[0] for f in chunk)
            frames = torch.zeros(len(chunk), T, chunk[0].shape[1])
            mask = torch.zeros(len(chunk), T, dtype=torch.bool)
            for j, f in enumerate(chunk):
                frames[j, :f.shape[0]] = torch.from_numpy(f)
                mask[j, :f.shape[0]] = True
            emb = head(frames.to(device), mask.to(device))
            out.append(F.normalize(emb, dim=-1).cpu())
    return torch.cat(out)


def eval_cross_domain(head, eval_pack, device):
    """Zero-shot TTS-centroid vs real-clip scoring on held-out words."""
    embs = {}
    for w, pack in eval_pack.items():
        embs[w] = {"tts": embed_features(head, pack["tts"], device),
                   "real": embed_features(head, pack["real"], device)}
    pos, neg = [], []
    words = sorted(embs)
    for w in words:
        centroid = F.normalize(embs[w]["tts"].mean(dim=0), dim=-1)
        pos.extend((embs[w]["real"] @ centroid).tolist())
        for v in words:
            if v != w:
                neg.extend((embs[v]["real"] @ centroid).tolist())
    pos, neg = np.array(pos), np.array(neg)
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    auc = (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) \
        / (len(pos) * len(neg))
    return float(auc), float(pos.mean()), float(neg.mean())


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    print(f"Phase-3 attentive-head training on {device}")

    ds = CachedWordDataset(args.cache_dir)
    sampler = ConfusableBatchSampler(
        ds, classes_per_batch=args.classes_per_batch,
        samples_per_class=args.samples_per_class,
        batches_per_epoch=args.batches_per_epoch,
        p_confusable=args.p_confusable, seed=args.seed)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_pad,
                        num_workers=args.num_workers)
    eval_pack = ds.eval_pack()
    print(f"Eval: {len(eval_pack)} zero-shot words.")

    dim = ds.features(ds.words[0], 0).shape[1]
    head = AttentivePoolingHead(dim=dim, n_heads=args.n_heads).to(device)
    loss_fn = SubCenterArcFace(dim, len(ds.words), k=args.subcenters,
                               s=args.arc_scale, m=args.arc_margin).to(device)

    auc0, pos0, neg0 = eval_cross_domain(head, eval_pack, device)
    print(f"Epoch 0 (identity init == wavlm-L10 mean pool): "
          f"AUC {auc0:.4f}  pos-cos {pos0:.3f}  neg-cos {neg0:.3f}")
    save_head_checkpoint(head, "siamese_v3_best.pth",
                         {"layer": ds.meta["layer"],
                          "backbone": ds.meta["backbone"],
                          "epoch": 0, "auc": auc0})

    opt = torch.optim.AdamW([
        {"params": head.parameters(), "lr": args.lr},
        {"params": loss_fn.parameters(), "lr": args.lr * 10}],
        weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    extra = {"layer": None, "backbone": None}
    extra["layer"] = ds.meta["layer"]; extra["backbone"] = ds.meta["backbone"]

    best_auc, best_epoch = auc0, 0
    for epoch in range(1, args.epochs + 1):
        head.train()
        total, n = 0.0, 0
        for frames, mask, labels, _domains in loader:
            frames, mask = frames.to(device), mask.to(device)
            labels = labels.to(device)
            opt.zero_grad()
            emb = head(frames, mask)
            loss = loss_fn(emb, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(head.parameters()) + list(loss_fn.parameters()), 5.0)
            opt.step()
            total += loss.item(); n += 1
        sched.step()
        head.eval()
        auc, pos, neg = eval_cross_domain(head, eval_pack, device)
        print(f"--- Epoch {epoch}: arcface {total / max(1, n):.4f} | "
              f"AUC {auc:.4f} | pos-cos {pos:.3f} neg-cos {neg:.3f} "
              f"(gap {pos - neg:+.3f}) ---")
        if auc > best_auc:
            best_auc, best_epoch = auc, epoch
            save_head_checkpoint(head, "siamese_v3_best.pth",
                                 {**extra, "epoch": epoch, "auc": auc})
            print(f"New best AUC {auc:.4f} -> siamese_v3_best.pth")

    save_head_checkpoint(head, "siamese_v3_final.pth",
                         {**extra, "epoch": args.epochs, "auc": auc})
    print(f"Done. Best AUC {best_auc:.4f} at epoch {best_epoch} "
          f"(epoch-0 mean-pool baseline {auc0:.4f}).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase-3 attentive-head training.")
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--classes-per-batch", type=int, default=16)
    ap.add_argument("--samples-per-class", type=int, default=4)
    ap.add_argument("--batches-per-epoch", type=int, default=250)
    ap.add_argument("--p-confusable", type=float, default=0.5)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--subcenters", type=int, default=3)
    ap.add_argument("--arc-scale", type=float, default=30.0)
    ap.add_argument("--arc-margin", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    train(ap.parse_args())
