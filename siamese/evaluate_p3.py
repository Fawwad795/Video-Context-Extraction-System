"""P3 (part A) - robustness & retrieval evaluation of a trained Siamese checkpoint.

Two deployment-relevant stress tests on the held-out TEST split, run on CPU with the trained
checkpoint (no GPU needed):

  1. Noise robustness: add background noise at decreasing SNRs (clean -> 0 dB) to BOTH clips of
     each pair (a conservative ablation) and report AUC + accuracy-at-clean-threshold vs SNR.
  2. Retrieval mAP: embed many test clips, rank every clip against a query by cosine similarity,
     and measure how well same-word clips rank at the top (the thesis's metric).

    python -m siamese.evaluate_p3 --ckpt siamese/checkpoints/siamese_full_v1.pt
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from .audio import log_mel
from .augment import add_background_noise
from .baseline_correlation import best_threshold
from .config import ARTIFACTS_DIR
from .datasets import PairDataset, SpeechCommandsWords
from .model import build_encoder
from .train import P0_AUC

SNRS = [None, 20, 15, 10, 5, 0]   # None = clean


@torch.no_grad()
def _embed_indices(model, base, indices, device, snr=None, batch=256):
    """Map each index -> embedding, optionally adding background noise at `snr` dB first."""
    embs, specs, keys = {}, [], []

    def flush():
        if specs:
            out = model.embed(torch.stack(specs).to(device)).cpu()
            for k, e in zip(keys, out):
                embs[k] = e
            specs.clear()
            keys.clear()

    for i in indices:
        w = base.raw_waveform(i)
        if snr is not None:
            w = add_background_noise(w, snr_db=snr)
        specs.append(log_mel(w))
        keys.append(i)
        if len(specs) >= batch:
            flush()
    flush()
    return embs


def robustness(model, base, pairs, device):
    uniq = sorted({i for p in pairs for i in p[:2]})
    labels = np.array([y for _, _, y in pairs])
    results, clean_thr = {}, None
    for snr in SNRS:
        embs = _embed_indices(model, base, uniq, device, snr=snr)
        scores = np.array([float((embs[a] * embs[b]).sum()) for a, b, _ in pairs])
        auc = roc_auc_score(labels, scores)
        if snr is None:                       # calibrate the threshold on the clean condition
            clean_thr, _ = best_threshold(scores, labels)
        acc = accuracy_score(labels, scores >= clean_thr)
        results[snr] = (auc, acc)
    return results, clean_thr


def retrieval_map(model, base, indices, device):
    embs = _embed_indices(model, base, indices, device)
    E = torch.stack([embs[i] for i in indices])        # (N, d) unit-norm
    labels = np.array([base.label_of(i) for i in indices])
    S = (E @ E.t()).numpy()
    np.fill_diagonal(S, -np.inf)                        # exclude self-match
    aps = []
    for q in range(len(indices)):
        order = np.argsort(-S[q])
        rel = (labels[order] == labels[q]).astype(int)
        if rel.sum() == 0:
            continue
        cum = np.cumsum(rel)
        prec = cum / (np.arange(len(rel)) + 1)
        aps.append(float((prec * rel).sum() / rel.sum()))
    return float(np.mean(aps))


def _sample_indices(base, per_word=40):
    by_label = defaultdict(list)
    for i, lab in enumerate(base.labels):
        if lab != "_background_noise_":
            by_label[lab].append(i)
    out = []
    for idxs in by_label.values():
        out.extend(idxs[:per_word])
    return out


def main():
    ap = argparse.ArgumentParser(description="P3 robustness + retrieval eval.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-pairs", type=int, default=2000)
    ap.add_argument("--map-per-word", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = build_encoder(ckpt["model"]).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    print(f"checkpoint: {args.ckpt} (model={ckpt['model']}, device={device})")

    test = SpeechCommandsWords("testing")
    pairs = PairDataset(test, neg_per_pos=1, max_pairs=args.n_pairs, seed=args.seed).pairs

    print(f"\n[1/2] Noise robustness on {len(pairs)} pairs (noise added to BOTH clips)...")
    rob, thr = robustness(model, test, pairs, device)
    print(f"  clean-calibrated cosine threshold: {thr:.3f}")
    print(f"  {'SNR':>7} | {'AUC':>6} | {'acc@thr':>7}")
    for snr in SNRS:
        auc, acc = rob[snr]
        tag = "clean" if snr is None else f"{snr} dB"
        print(f"  {tag:>7} | {auc:6.3f} | {acc:7.3f}")

    print(f"\n[2/2] Retrieval mAP ({args.map_per_word} clips/word)...")
    idx = _sample_indices(test, per_word=args.map_per_word)
    mapv = retrieval_map(model, test, idx, device)
    print(f"  clips: {len(idx)} | mAP: {mapv:.3f}")

    # ---- figure: robustness curves ----
    xs = [30 if s is None else s for s in SNRS]          # clean plotted at 30 dB
    aucs = [rob[s][0] for s in SNRS]
    accs = [rob[s][1] for s in SNRS]
    fig, axx = plt.subplots(figsize=(7, 4.3))
    axx.plot(xs, aucs, "o-", label="Siamese AUC")
    axx.plot(xs, accs, "s--", label="Siamese acc @ clean thr")
    axx.axhline(P0_AUC, color="gray", ls=":", label=f"P0 correlation AUC ({P0_AUC})")
    axx.set_xlabel("SNR (dB)  —  left = noisier, 'clean' at 30")
    axx.set_ylabel("score")
    axx.set_ylim(0.45, 1.0)
    axx.invert_xaxis()
    axx.set_title(f"P3 noise robustness ({ckpt['model']} model) | mAP={mapv:.3f}")
    axx.legend(loc="lower left")
    fig.tight_layout()
    out = ARTIFACTS_DIR / f"p3_robustness_{ckpt['model']}.png"
    fig.savefig(out, dpi=120)
    print(f"\n  saved figure: {out}")
    print("P3 part A done.")


if __name__ == "__main__":
    main()
