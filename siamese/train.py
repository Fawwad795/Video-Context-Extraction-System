"""Train + evaluate the Siamese keyword matcher (roadmap phase P2).

Trains the shared CNN encoder with contrastive loss on Speech Commands pairs, calibrates a
decision threshold on validation, and reports held-out TEST AUC/accuracy/F1 against the P0
correlation baseline (AUC 0.555 / acc 0.54).

Examples
--------
CPU smoke test (correctness only):
    python -m siamese.train --model reduced --epochs 1 \
        --max-train-pairs 256 --max-eval-pairs 256 --workers 0 --device cpu

Full GPU run (on the AWS box):
    python -m siamese.train --model full --epochs 15 --batch-size 256 --workers 8
"""
import argparse
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from .config import CHECKPOINT_DIR
from .datasets import PairDataset, SpeechCommandsWords
from .model import build_encoder, contrastive_loss, cosine_score

# P0 baseline numbers (siamese/baseline_correlation.py) for side-by-side reporting.
P0_AUC, P0_ACC = 0.555, 0.540


def make_loader(subset, neg_per_pos, max_pairs, batch_size, seed, shuffle, workers):
    base = SpeechCommandsWords(subset)
    pairs = PairDataset(base, neg_per_pos=neg_per_pos, max_pairs=max_pairs, seed=seed)
    return DataLoader(pairs, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=(workers > 0))


@torch.no_grad()
def evaluate(model, loader, device):
    """Return AUC + best-threshold accuracy/F1 over a loader of pairs."""
    model.eval()
    scores, labels = [], []
    for a, b, y in loader:
        e1, e2 = model(a.to(device), b.to(device))
        scores.append(cosine_score(e1, e2).cpu().numpy())
        labels.append(y.numpy())
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    auc = roc_auc_score(labels, scores)
    _, _, thr = roc_curve(labels, scores)
    thr = thr[np.isfinite(thr)]
    acc, best_t = max((accuracy_score(labels, scores >= t), float(t)) for t in thr)
    f1 = f1_score(labels, scores >= best_t)
    # cast to plain floats so checkpoints stay weights_only-safe
    return dict(auc=float(auc), acc=float(acc), f1=float(f1), threshold=float(best_t))


def main():
    ap = argparse.ArgumentParser(description="Train the Siamese keyword matcher.")
    ap.add_argument("--model", choices=["full", "reduced"], default="full")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--neg-per-pos", type=int, default=2)
    ap.add_argument("--max-train-pairs", type=int, default=None)
    ap.add_argument("--max-eval-pairs", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"device={device} | model={args.model} | epochs={args.epochs} | bs={args.batch_size}")

    train_loader = make_loader("training", args.neg_per_pos, args.max_train_pairs,
                               args.batch_size, args.seed, True, args.workers)
    val_loader = make_loader("validation", 1, args.max_eval_pairs,
                             args.batch_size, args.seed, False, args.workers)
    test_loader = make_loader("testing", 1, args.max_eval_pairs,
                              args.batch_size, args.seed, False, args.workers)
    print(f"pairs -> train {len(train_loader.dataset)} | "
          f"val {len(val_loader.dataset)} | test {len(test_loader.dataset)}")

    model = build_encoder(args.model).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_auc = 0.0
    ckpt_path = CHECKPOINT_DIR / f"siamese_{args.model}_{args.tag}.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for a, b, y in train_loader:
            a, b, y = a.to(device), b.to(device), y.to(device)
            e1, e2 = model(a, b)
            loss = contrastive_loss(e1, e2, y, margin=args.margin)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            n += len(y)
        val = evaluate(model, val_loader, device)
        print(f"epoch {epoch:2d} | loss {total / n:.4f} | "
              f"val AUC {val['auc']:.3f} acc {val['acc']:.3f} | {time.time() - t0:.0f}s")
        if val["auc"] > best_auc:
            best_auc = val["auc"]
            torch.save({"state_dict": model.state_dict(), "model": args.model,
                        "args": vars(args), "val": val}, ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test = evaluate(model, test_loader, device)
    print("\n==== P2 Siamese result (held-out TEST) ====")
    print(f"  best val AUC          : {best_auc:.3f}")
    print(f"  TEST AUC {test['auc']:.3f} | acc {test['acc']:.3f} | "
          f"F1 {test['f1']:.3f} | thr {test['threshold']:.3f}")
    print(f"  P0 correlation baseline: AUC {P0_AUC:.3f} / acc {P0_ACC:.3f}")
    print(f"  -> AUC lift: +{test['auc'] - P0_AUC:.3f}")
    print(f"  saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
