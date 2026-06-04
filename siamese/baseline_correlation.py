"""P0 - reproduce the VMS cross-correlation detector as a pair-similarity baseline.

The original VMS scores a match with a normalized cross-correlation "matching percentage"
(Research/Stream1_corelation_updated_v2.py, lines 71-88) and fires above a hand-tuned 70%.
Here we apply that exact score to the held-out Speech Commands TEST pairs, calibrate the
threshold on VALIDATION, and report accuracy + ROC-AUC. This establishes the number the
Siamese model must beat in P3.

Run from the repo root:   python -m siamese.baseline_correlation

Notes:
- We use scipy's FFT-based correlation, which is numerically identical to np.correlate but
  fast enough to score thousands of 1 s clips.
- The report's 72% was measured on TIMIT (a licensed corpus we don't have), so it is not
  directly comparable; this gives an internally consistent baseline on the SAME test pairs
  the Siamese model will be evaluated on.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve

from .config import ARTIFACTS_DIR
from .datasets import PairDataset, SpeechCommandsWords


def matching_percentage(y: np.ndarray, w: np.ndarray) -> float:
    """Normalized cross-correlation matching %, faithful to the original VMS detector."""
    corr = correlate(y, w, mode="same", method="fft")
    idx = int(np.argmax(corr))
    match = y[idx: idx + len(w)]
    denom = (np.linalg.norm(match) * np.linalg.norm(w)) + 1e-12
    return min(float(np.max(corr) / denom) * 100.0, 100.0)


def score_pairs(base: SpeechCommandsWords, pairs):
    """Return (scores, labels) arrays for a list of (idx_a, idx_b, label) pairs."""
    cache: dict[int, np.ndarray] = {}

    def wav(i: int) -> np.ndarray:
        if i not in cache:
            cache[i] = base.raw_waveform(i).numpy()
        return cache[i]

    scores = np.empty(len(pairs), dtype=np.float64)
    labels = np.empty(len(pairs), dtype=np.int64)
    for k, (a, b, y) in enumerate(pairs):
        scores[k] = matching_percentage(wav(a), wav(b))
        labels[k] = y
    return scores, labels


def best_threshold(scores: np.ndarray, labels: np.ndarray):
    """Threshold (and its accuracy) that maximizes accuracy over an ROC sweep."""
    _, _, thr = roc_curve(labels, scores)
    thr = thr[np.isfinite(thr)]
    best_t, best_acc = 50.0, 0.0
    for t in thr:
        acc = accuracy_score(labels, scores >= t)
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t, best_acc


def _plot(test_scores, test_labels, thr, auc, out_path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    # ROC
    fpr, tpr, _ = roc_curve(test_labels, test_scores)
    ax[0].plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate",
              title="Correlation baseline ROC (test)")
    ax[0].legend(loc="lower right")
    # score distributions
    pos = test_scores[test_labels == 1]
    neg = test_scores[test_labels == 0]
    bins = np.linspace(0, 100, 40)
    ax[1].hist(neg, bins=bins, alpha=0.6, label="different word", density=True)
    ax[1].hist(pos, bins=bins, alpha=0.6, label="same word", density=True)
    ax[1].axvline(thr, color="k", ls="--", lw=1, label=f"threshold {thr:.1f}%")
    ax[1].set(xlabel="matching %", ylabel="density", title="Score distributions (test)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)


def main(n_val: int = 2000, n_test: int = 4000, seed: int = 1):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    val = SpeechCommandsWords("validation")
    test = SpeechCommandsWords("testing")
    val_pairs = PairDataset(val, neg_per_pos=1, max_pairs=n_val, seed=seed).pairs
    test_pairs = PairDataset(test, neg_per_pos=1, max_pairs=n_test, seed=seed).pairs

    print(f"Scoring {len(val_pairs)} val + {len(test_pairs)} test pairs with np.correlate ...")
    vs, vl = score_pairs(val, val_pairs)
    ts, tl = score_pairs(test, test_pairs)

    thr, val_acc = best_threshold(vs, vl)
    pred = ts >= thr
    test_acc = accuracy_score(tl, pred)
    test_f1 = f1_score(tl, pred)
    test_auc = roc_auc_score(tl, ts)
    acc70 = accuracy_score(tl, ts >= 70.0)

    print("\n==== P0 correlation baseline (held-out TEST) ====")
    print(f"  pairs: val={len(vl)} (pos {int(vl.sum())}) | test={len(tl)} (pos {int(tl.sum())})")
    print(f"  calibrated threshold (from val): {thr:.2f}%   [val acc {val_acc:.3f}]")
    print(f"  TEST accuracy @ calibrated thr : {test_acc:.3f}")
    print(f"  TEST F1       @ calibrated thr : {test_f1:.3f}")
    print(f"  TEST accuracy @ fixed 70%      : {acc70:.3f}")
    print(f"  TEST ROC-AUC (threshold-free)  : {test_auc:.3f}")

    out = ARTIFACTS_DIR / "p0_correlation_baseline.png"
    _plot(ts, tl, thr, test_auc, out)
    print(f"\n  saved figure: {out}")
    print("P0 baseline done.")
    return {"threshold": thr, "test_acc": test_acc, "test_f1": test_f1,
            "test_auc": test_auc, "acc@70": acc70}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="P0 cross-correlation baseline on Speech Commands.")
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    main(n_val=args.n_val, n_test=args.n_test, seed=args.seed)
