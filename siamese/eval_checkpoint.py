"""Evaluate a saved Siamese checkpoint on the held-out TEST split.

Decoupled from training so we can re-score a checkpoint without retraining (and so the P2
test number can be produced even if a training run's final eval was interrupted).

    python -m siamese.eval_checkpoint --ckpt siamese/checkpoints/siamese_full_v1.pt
"""
from __future__ import annotations

import argparse

import torch

from .model import build_encoder
from .train import P0_ACC, P0_AUC, evaluate, make_loader


def main():
    ap = argparse.ArgumentParser(description="Evaluate a Siamese checkpoint on the test split.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max-eval-pairs", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = build_encoder(ckpt["model"]).to(device)
    model.load_state_dict(ckpt["state_dict"])

    test = make_loader("testing", 1, args.max_eval_pairs, 256, args.seed, False, args.workers)
    r = evaluate(model, test, device)

    print(f"checkpoint: {args.ckpt}  (model={ckpt['model']})")
    print(f"best val : auc {ckpt['val']['auc']:.3f} acc {ckpt['val']['acc']:.3f}")
    print("==== TEST (held-out) ====")
    print(f"  Siamese : AUC {r['auc']:.3f} | acc {r['acc']:.3f} | F1 {r['f1']:.3f} | thr {r['threshold']:.3f}")
    print(f"  P0 corr : AUC {P0_AUC:.3f} | acc {P0_ACC:.3f}")
    print(f"  lift    : +{r['auc'] - P0_AUC:.3f} AUC / +{r['acc'] - P0_ACC:.3f} acc")


if __name__ == "__main__":
    main()
