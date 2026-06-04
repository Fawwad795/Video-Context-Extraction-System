"""P4.5 training - fine-tune with continuous-speech hard negatives to fix over-detection.

Warm-starts from the P2 checkpoint and continues training on ContinuousPairDataset (which adds
boundary/fragment "junk" windows as hard negatives). Reports, before vs after:
  - isolated-word test AUC (should stay high, ~0.98), and
  - continuous-speech FPR/TPR at a fixed threshold (FPR should drop sharply).

Example (GPU):
    python -m siamese.train_p45 --init-ckpt siamese/checkpoints/siamese_full_v1.pt \
        --model full --epochs 12 --n-train-pairs 60000 --workers 8
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import CHECKPOINT_DIR, DATA_DIR
from .continuous import (ContinuousPairDataset, DenseSpeechWindows,
                         clean_prototype, continuous_fpr_tpr)
from .datasets import SpeechCommandsWords
from .model import build_encoder, contrastive_loss
from .train import evaluate, make_loader

# Sample of Speech Commands words used to probe continuous-speech FPR/TPR.
EVAL_KEYWORDS = ["right", "stop", "yes", "seven", "house"]


def continuous_report(model, base, dense, threshold, device, n_neg=600):
    fprs, tprs = [], []
    for kw in EVAL_KEYWORDS:
        proto = clean_prototype(model, base, kw, k=10, device=device)
        r = continuous_fpr_tpr(model, base, dense, proto, kw, threshold, n_neg=n_neg, device=device)
        fprs.append(r["fpr"])
        tprs.append(r["tpr"])
    return float(np.mean(fprs)), float(np.mean(tprs))


def main():
    ap = argparse.ArgumentParser(description="P4.5 continuous-speech hard-negative fine-tuning.")
    ap.add_argument("--init-ckpt", default=None, help="warm-start checkpoint (e.g. siamese_full_v1.pt)")
    ap.add_argument("--model", choices=["full", "reduced"], default="full")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--n-train-pairs", type=int, default=60000)
    ap.add_argument("--max-eval-pairs", type=int, default=4000)
    ap.add_argument("--threshold", type=float, default=0.80)
    ap.add_argument("--librispeech", default=str(DATA_DIR / "LibriSpeech" / "dev-clean"),
                    help="dir of LibriSpeech .flac files used as dense-speech hard negatives")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="p45")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"device={device} | model={args.model} | hard-negative fine-tune", flush=True)

    dense = DenseSpeechWindows(args.librispeech)
    print(f"dense-speech negatives: {len(dense.files)} LibriSpeech files", flush=True)
    train_base = SpeechCommandsWords("training")
    val_base = SpeechCommandsWords("validation")
    test_base = SpeechCommandsWords("testing")
    train_ds = ContinuousPairDataset(train_base, dense, n_pairs=args.n_train_pairs, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=(args.workers > 0))
    val_iso = make_loader("validation", 1, args.max_eval_pairs, args.batch_size, args.seed, False, args.workers)
    test_iso = make_loader("testing", 1, args.max_eval_pairs, args.batch_size, args.seed, False, args.workers)

    model = build_encoder(args.model).to(device)
    if args.init_ckpt:
        ck = torch.load(args.init_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        print(f"warm-started from {args.init_ckpt}", flush=True)

    model.eval()
    f0, t0 = continuous_report(model, test_base, dense, args.threshold, device, n_neg=2000)
    print(f"[before] TEST continuous FPR {f0:.3f} | TPR {t0:.3f} @ thr {args.threshold}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_score, ckpt_path = -1.0, CHECKPOINT_DIR / f"siamese_{args.model}_{args.tag}.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        t, tot, n = time.time(), 0.0, 0
        for a, b, y in train_loader:
            a, b, y = a.to(device), b.to(device), y.to(device)
            e1, e2 = model(a, b)
            loss = contrastive_loss(e1, e2, y, margin=args.margin)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
            n += len(y)
        iso = evaluate(model, val_iso, device)
        fpr, tpr = continuous_report(model, val_base, dense, args.threshold, device)
        score = tpr - fpr                      # want high TPR, low FPR
        print(f"epoch {epoch:2d} | loss {tot / n:.4f} | iso AUC {iso['auc']:.3f} | "
              f"cont FPR {fpr:.3f} TPR {tpr:.3f} | {time.time() - t:.0f}s", flush=True)
        if score > best_score:
            best_score = score
            torch.save({"state_dict": model.state_dict(), "model": args.model,
                        "args": vars(args)}, ckpt_path)

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    iso = evaluate(model, test_iso, device)
    f1, t1 = continuous_report(model, test_base, dense, args.threshold, device, n_neg=2000)
    print("\n==== P4.5 result (held-out TEST) ====", flush=True)
    print(f"  isolated-word AUC : {iso['auc']:.3f}  (P2 full = 0.987; should stay high)")
    print(f"  continuous FPR    : {f0:.3f} -> {f1:.3f}   (lower is better)")
    print(f"  continuous TPR    : {t0:.3f} -> {t1:.3f}   (keep high)")
    print(f"  saved: {ckpt_path}")


if __name__ == "__main__":
    main()
