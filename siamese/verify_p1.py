"""P1 sanity check — proves the data pipeline end-to-end on real data.

Run from the repo root:   python -m siamese.verify_p1

Reports split sizes, pair label balance, and batch tensor shapes, and saves a figure
comparing clean / noisy / SpecAugment spectrograms plus a positive and negative pair.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from . import augment
from .audio import load_wav, log_mel
from .config import ARTIFACTS_DIR, NOISE_LABEL
from .datasets import PairDataset, SpeechCommandsWords


def _show(ax, title, spec):
    ax.imshow(spec.squeeze(0).numpy(), origin="lower", aspect="auto")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("== Speech Commands subsets (official splits) ==")
    splits = {}
    for s in ("training", "validation", "testing"):
        ds = SpeechCommandsWords(s)
        splits[s] = ds
        words = sorted({l for l in ds.labels if l != NOISE_LABEL})
        print(f"  {s:10s}: {len(ds):6d} clips | {len(words):2d} words")

    val = splits["validation"]
    pairs = PairDataset(val, neg_per_pos=1, max_pairs=400, seed=1)
    ys = [y for _, _, y in pairs.pairs]
    print(f"\n== Pairs (validation, capped 400): {len(pairs)} | "
          f"positives={sum(ys)} negatives={len(ys) - sum(ys)} ==")

    loader = DataLoader(pairs, batch_size=16, shuffle=True)
    a, b, y = next(iter(loader))
    print(f"== Batch: A={tuple(a.shape)} B={tuple(b.shape)} y={tuple(y.shape)} | "
          f"spec min={a.min():.2f} max={a.max():.2f} mean={a.mean():.2f} ==")

    # ---- figure: augmentation + a positive/negative pair ----
    relpath, _, label, spk, _ = val._metas[0]
    wav, _ = load_wav(os.path.join(val._base, relpath))
    clean = log_mel(wav)
    noisy = log_mel(augment.add_background_noise(wav, snr_db=5.0))
    auged = augment.spec_augment(clean)

    pos = next((p for p in pairs.pairs if p[2] == 1), None)
    neg = next((p for p in pairs.pairs if p[2] == 0), None)

    fig, ax = plt.subplots(2, 3, figsize=(12, 6))
    _show(ax[0, 0], f"clean '{label}'", clean)
    _show(ax[0, 1], "+ background noise (5 dB)", noisy)
    _show(ax[0, 2], "+ SpecAugment", auged)
    _show(ax[1, 0], f"pos A: '{val.label_of(pos[0])}'", val[pos[0]][0])
    _show(ax[1, 1], f"pos B: '{val.label_of(pos[1])}'", val[pos[1]][0])
    _show(ax[1, 2], f"neg B: '{val.label_of(neg[1])}'", val[neg[1]][0])
    fig.suptitle("P1 pipeline check — log-mel spectrograms")
    fig.tight_layout()
    out = ARTIFACTS_DIR / "p1_pipeline_check.png"
    fig.savefig(out, dpi=110)
    print(f"\n== Saved figure: {out} ==")
    print("P1 pipeline OK.")


if __name__ == "__main__":
    main()
