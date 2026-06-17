"""Stage B (run in 'udkws_torch' docker): score the VMS chunks with the trained
PhonMatchNet, using the Stage-A Google embeddings, for each text keyword.

Per window: forward((x_raw, gemb), g2p_embed(keyword)) -> detection prob.
Chunk score = max prob over its windows. Threshold swept for best F1 vs the
Whisper-derived ground truth.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home")
sys.path.insert(0, "/home/dataset")
from model import ukws                       # noqa: E402
from g2p.g2p_en.g2p import G2p               # noqa: E402

SR, WIN, MAXLEN_A, FRAME, HOP = 16000, 24000, 32000, 400, 160
GEMB_NPZ = "/out/vms_gemb.npz"
CKPT = "/home/results/checkpoint/epoch_13/model.pt"   # best overall EER (torch.save)

# Ground truth for the current debate chunks (from Whisper word timestamps)
GROUND_TRUTH = {
    "penalty": {"live_4.wav", "live_7.wav"},
    "elon":    {"live_0.wav"},
}


def gemb_len_of(x_len):
    return int(int((x_len - FRAME) / HOP + 1) / 8)


MAXLEN_GEMB = gemb_len_of(MAXLEN_A)


def load_model():
    kwargs = dict(vocab=72, text_input="g2p_embed", audio_input="both",
                  stack_extractor=True, frame_length=FRAME, hop_length=HOP,
                  num_mel=40, sample_rate=SR, log_mel=False)
    model = ukws.BaseUKWS(**kwargs)
    sd = torch.load(CKPT, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded {CKPT} (missing={len(missing)}, unexpected={len(unexpected)})")
    if missing:
        print("  missing:", missing)
    model.eval()
    return model


def best_f1(rows):
    """rows = [(name, truth_bool, score)]. Sweep threshold for best F1."""
    scored = sorted(set(r[2] for r in rows))
    best = (0.0, None, 0, 0, 0, 0)
    for t in scored + [scored[-1] + 1e-6]:
        tp = sum(1 for n, tr, s in rows if tr and s >= t)
        fp = sum(1 for n, tr, s in rows if (not tr) and s >= t)
        fn = sum(1 for n, tr, s in rows if tr and s < t)
        tn = sum(1 for n, tr, s in rows if (not tr) and s < t)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if f1 > best[0]:
            best = (f1, t, tp, fp, fn, tn)
    return best


def main():
    model = load_model()
    g2p = G2p()
    data = np.load(GEMB_NPZ)
    chunks = sorted({k.split("__")[0] for k in data.files},
                    key=lambda x: int(x.split("_")[1].split(".")[0]))

    for kw, gt in GROUND_TRUTH.items():
        y = torch.tensor(np.array(g2p.embedding(kw)), dtype=torch.float32)[None]  # (1,P,256)
        y_len = torch.tensor([y.shape[1]], dtype=torch.int32)
        print(f"\n===== keyword '{kw}'  (g2p_embed {tuple(y.shape)}) =====")
        rows = []
        for name in chunks:
            wins, gembs = data[name + "__win"], data[name + "__gemb"]
            probs = []
            for i in range(len(wins)):
                x = torch.tensor(np.pad(wins[i], (0, MAXLEN_A - len(wins[i]))),
                                 dtype=torch.float32)[None]
                with torch.no_grad():
                    # match gemb length to the model's own mel-frame count (//8)
                    mel, _ = model.SPEC(x)
                    target = mel.shape[1] // 8
                    gm = gembs[i]
                    gm = gm[:target] if gm.shape[0] >= target else \
                        np.pad(gm, ((0, target - gm.shape[0]), (0, 0)))
                    gm = torch.tensor(gm, dtype=torch.float32)[None]
                    x_len = torch.tensor([WIN], dtype=torch.int32)
                    g_len = torch.tensor([min(gemb_len_of(WIN), target)], dtype=torch.int32)
                    prob = model((x, gm), y, (x_len, g_len), y_len)[0]
                probs.append(float(prob.reshape(-1)[0]))
            score = max(probs)
            truth = name in gt
            rows.append((name, truth, score))
            print(f"  {name}  truth={str(truth):<5} score={score:.3f}")
        f1, t, tp, fp, fn, tn = best_f1(rows)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        print(f"  -> best F1={f1:.2f} @thr={t:.3f}  P={prec:.2f} R={rec:.2f}  "
              f"(TP={tp} FP={fp} FN={fn} TN={tn})")


if __name__ == "__main__":
    main()
