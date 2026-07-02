"""Phase-2 training: TTS-mixed triplets + gradient-reversal domain classifier.

Loss (per the Google Interspeech 2024 recipe, arXiv:2408.10463):

    L = (1 - beta) * TripletMarginLoss + beta * BCE(domain | GRL(embedding))

The gradient reversal layer (GRL) rewards the projection head for making
real and synthetic embeddings indistinguishable to the domain classifier,
which is exactly the synthetic-to-real gap that breaks TTS anchors at
inference. beta ramps 0 -> beta_max over the first beta_ramp epochs;
gradients are clipped (GRLs destabilize training otherwise).

Validation metric: cross-domain AUC on held-out eval words - a 3-voice TTS
centroid per word scored by cosine against real human clips of the same
word (positives) and of other words (negatives). This is the deployment
scenario (TTS prototype anchor vs live human speech), measured zero-shot
on words never seen in training triplets.

Checkpoints: siamese_v2_best.pth (best AUC), siamese_v2_epoch_<n>.pth,
siamese_v2_final.pth. The v1 checkpoint best_siamese_model.pth is used as
warm start and never overwritten.
"""

import argparse
import os
import random

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import rankdata
from torch.optim.lr_scheduler import CosineAnnealingLR

# Shared modules (scoring, siamese_model, augment_utils) live in ../core
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "core"))

from dataset_v2 import CLIP_SAMPLES, SR, SpeechTripletDomainDataset, make_dataloader
from siamese_model import SiameseAudioModel


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def grad_reverse(x):
    return GradReverse.apply(x)


def pooled_features(model, clips, batch_size=16):
    """Frozen-backbone mean-pooled 768-d features (cacheable for eval)."""
    device = next(model.parameters()).device
    feats = []
    with torch.no_grad():
        for i in range(0, len(clips), batch_size):
            batch = [np.asarray(c, dtype=np.float32) for c in clips[i:i + batch_size]]
            inputs = model.feature_extractor(batch, sampling_rate=SR,
                                             return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            hidden = model.backbone(**inputs).last_hidden_state
            feats.append(hidden.mean(dim=1))
    return torch.cat(feats, dim=0)


def build_eval(model, ds, n_words=150, pos_per_word=2, neg_per_word=6, seed=123):
    """Cache backbone features for the cross-domain eval set (computed once -
    only the projection head changes between epochs)."""
    rng = random.Random(seed)
    words = [w for w in ds.eval_words
             if w in ds.eval_bank and len(ds.eval_class_to_indices[w]) >= pos_per_word]
    rng.shuffle(words)
    words = words[:n_words]
    if len(words) < 10:
        raise RuntimeError("Too few usable eval words - check the TTS bank manifest.")

    clips, tts_rows, pos_rows = [], {}, {}
    for w in words:
        rows = []
        for p in ds.eval_bank[w][:3]:
            audio, _ = librosa.load(p, sr=SR)
            clips.append(ds.fix_length(audio))
            rows.append(len(clips) - 1)
        tts_rows[w] = rows
        rows = []
        for idx in rng.sample(ds.eval_class_to_indices[w], pos_per_word):
            clips.append(ds.get_real_clip(idx))
            rows.append(len(clips) - 1)
        pos_rows[w] = rows

    # Fixed negative assignments: real clips of OTHER eval words
    neg_rows = {}
    for w in words:
        others = [r for w2 in words if w2 != w for r in pos_rows[w2]]
        neg_rows[w] = rng.sample(others, min(neg_per_word, len(others)))

    print(f"Eval set: {len(words)} zero-shot words, {len(clips)} clips "
          f"(backbone features cached once).")
    feats = pooled_features(model, clips)
    return {"words": words, "feats": feats, "tts_rows": tts_rows,
            "pos_rows": pos_rows, "neg_rows": neg_rows}


def rank_auc(pos_scores, neg_scores):
    scores = np.concatenate([pos_scores, neg_scores])
    ranks = rankdata(scores)
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    u = ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def eval_cross_domain_auc(model, pack):
    with torch.no_grad():
        emb = F.normalize(model.projection_head(pack["feats"]), dim=-1)
    pos, neg = [], []
    for w in pack["words"]:
        centroid = F.normalize(emb[pack["tts_rows"][w]].mean(dim=0), dim=-1)
        pos.extend((emb[pack["pos_rows"][w]] @ centroid).cpu().numpy().tolist())
        neg.extend((emb[pack["neg_rows"][w]] @ centroid).cpu().numpy().tolist())
    return rank_auc(np.array(pos), np.array(neg))


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Phase-2 GRL training on {device}")

    model = SiameseAudioModel()
    if args.init and os.path.exists(args.init):
        print(f"Warm-starting projection head from {args.init}")
        model.load_weights(args.init)
    model.to(device)
    model.projection_head.train()

    domain_head = nn.Sequential(
        nn.Linear(args.embed_dim, 64), nn.ReLU(), nn.Linear(64, 1)).to(device)

    triplet = nn.TripletMarginLoss(margin=1.0, p=2)
    bce = nn.BCEWithLogitsLoss()
    params = list(model.projection_head.parameters()) + list(domain_head.parameters())
    optimizer = optim.Adam(params, lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    ds, loader = make_dataloader(
        batch_size=args.batch_size, num_workers=args.num_workers,
        manifest_path=args.manifest, num_samples=args.num_samples,
        p_cross_domain=args.p_cross_domain, p_tts_negative=args.p_tts_negative)

    eval_pack = build_eval(model, ds, n_words=args.eval_words)
    baseline = eval_cross_domain_auc(model, eval_pack)
    print(f"Baseline cross-domain AUC (v1 head, zero-shot eval words): {baseline:.4f}")

    best_auc, best_epoch = baseline, 0
    for epoch in range(args.epochs):
        beta = args.beta_max * min(1.0, epoch / max(1, args.beta_ramp))
        trip_sum = dom_sum = dom_correct = dom_count = 0.0

        for batch_idx, (a, p, n, d_a, d_p, d_n) in enumerate(loader):
            a_np, p_np, n_np = list(a.numpy()), list(p.numpy()), list(n.numpy())

            optimizer.zero_grad()
            e_a = model.get_embedding(a_np, SR)
            e_p = model.get_embedding(p_np, SR)
            e_n = model.get_embedding(n_np, SR)

            trip_loss = triplet(e_a, e_p, e_n)
            embs = torch.cat([e_a, e_p, e_n], dim=0)
            labels = torch.cat([d_a, d_p, d_n], dim=0).float().unsqueeze(1).to(device)
            logits = domain_head(grad_reverse(embs))
            dom_loss = bce(logits, labels)

            loss = (1.0 - beta) * trip_loss + beta * dom_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()

            trip_sum += trip_loss.item()
            dom_sum += dom_loss.item()
            with torch.no_grad():
                dom_correct += ((torch.sigmoid(logits) > 0.5).float() == labels).sum().item()
                dom_count += labels.numel()

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch + 1}/{args.epochs} | Batch {batch_idx} | "
                      f"triplet {trip_loss.item():.4f} | domain {dom_loss.item():.4f} "
                      f"| beta {beta:.2f}")

        scheduler.step()
        auc = eval_cross_domain_auc(model, eval_pack)
        n_batches = max(1, len(loader))
        print(f"--- Epoch {epoch + 1}: triplet {trip_sum / n_batches:.4f} | "
              f"domain {dom_sum / n_batches:.4f} | "
              f"domain-acc {dom_correct / max(1, dom_count):.3f} | "
              f"cross-domain AUC {auc:.4f} ---")

        if auc > best_auc:
            best_auc, best_epoch = auc, epoch + 1
            model.save_weights("siamese_v2_best.pth")
            print(f"New best AUC {auc:.4f} -> siamese_v2_best.pth")
        if (epoch + 1) % 5 == 0:
            model.save_weights(f"siamese_v2_epoch_{epoch + 1}.pth")

    model.save_weights("siamese_v2_final.pth")
    print(f"Training complete. Best cross-domain AUC {best_auc:.4f} "
          f"at epoch {best_epoch} (baseline {baseline:.4f}).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase-2 GRL domain-adversarial training.")
    ap.add_argument("--manifest", default=os.path.expanduser("~/tts_bank/manifest.json"))
    ap.add_argument("--init", default="best_siamese_model.pth",
                    help="v1 checkpoint to warm-start from ('' = from scratch)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--num-samples", type=int, default=10000, help="triplets per epoch")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--beta-max", type=float, default=0.3)
    ap.add_argument("--beta-ramp", type=int, default=10, help="epochs to ramp beta")
    ap.add_argument("--grad-clip", type=float, default=0.5)
    ap.add_argument("--p-cross-domain", type=float, default=0.5)
    ap.add_argument("--p-tts-negative", type=float, default=0.3)
    ap.add_argument("--eval-words", type=int, default=150)
    train(ap.parse_args())
