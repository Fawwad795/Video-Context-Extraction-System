"""Portable PhonMatchNet keyword detector — runs locally, no docker.

Combines both inference stages in one process:
  Stage A (TensorFlow): Google Speech Embedding for 1.5 s sliding windows.
  Stage B (PyTorch):     PhonMatchNet scores each window for the text keyword.
A chunk's score is the max window probability; a threshold turns it into a
detection. Optionally compares to ground truth for F1.

Self-contained: the model code lives in ./phonmatchnet, the trained weights in
./phonmatchnet_model. Requires torch + tensorflow (see requirements-g2p.txt).

Examples:
    python detect.py --keyword penalty --audio-dir audios \
        --ground-truth live_4.wav,live_7.wav
    python detect.py --keyword elon --wav clip.wav --threshold 0.5
"""
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PM = os.path.join(HERE, "phonmatchnet")
sys.path.insert(0, PM)                                   # -> from model import ukws
sys.path.insert(0, os.path.join(PM, "dataset"))         # -> from g2p.g2p_en.g2p import G2p
sys.path.insert(0, os.path.join(PM, "google_speech_embedding"))  # -> speech_embedding

SR, WIN, HOP, MAXLEN_A, FRAME, HOPL = 16000, 24000, 8000, 32000, 400, 160
CKPT = os.path.join(HERE, "phonmatchnet_model", "phonmatchnet_epoch13.pt")


def gemb_len_of(x_len):
    return int(int((x_len - FRAME) / HOPL + 1) / 8)


def compute_gembs(chunks):
    """Stage A — Google Speech embeddings (TensorFlow) per sliding window."""
    from speech_embedding import GoogleSpeechEmbedder
    emb = GoogleSpeechEmbedder()
    out = {}
    for name, y in chunks.items():
        starts = list(range(0, max(1, len(y) - WIN + 1), HOP))
        wins, gembs = [], []
        for s in starts:
            w = y[s:s + WIN].astype(np.float32)
            if len(w) < WIN:
                w = np.pad(w, (0, WIN - len(w)))
            wp = np.pad(w, (0, MAXLEN_A - WIN))
            gembs.append(emb(wp[None, :]).numpy()[0])
            wins.append(w)
        out[name] = (np.stack(wins), np.stack(gembs))
    return out


def load_model():
    import torch
    from model import ukws
    m = ukws.BaseUKWS(vocab=72, text_input="g2p_embed", audio_input="both",
                      stack_extractor=True, frame_length=FRAME, hop_length=HOPL,
                      num_mel=40, sample_rate=SR, log_mel=False)
    sd = torch.load(CKPT, map_location="cpu")
    m.load_state_dict(sd)
    m.eval()
    return m


def score_chunks(model, gemb_data, keyword):
    """Stage B — PhonMatchNet detection probability per chunk (max over windows)."""
    import torch
    from g2p.g2p_en.g2p import G2p
    g2p = G2p()
    y = torch.tensor(np.array(g2p.embedding(keyword)), dtype=torch.float32)[None]
    y_len = torch.tensor([y.shape[1]], dtype=torch.int32)
    scores = {}
    for name, (wins, gembs) in gemb_data.items():
        probs = []
        for i in range(len(wins)):
            x = torch.tensor(np.pad(wins[i], (0, MAXLEN_A - len(wins[i]))),
                             dtype=torch.float32)[None]
            with torch.no_grad():
                mel, _ = model.SPEC(x)
                target = mel.shape[1] // 8
                gm = gembs[i]
                gm = gm[:target] if gm.shape[0] >= target else \
                    np.pad(gm, ((0, target - gm.shape[0]), (0, 0)))
                gm = torch.tensor(gm, dtype=torch.float32)[None]
                xl = torch.tensor([WIN], dtype=torch.int32)
                gl = torch.tensor([min(gemb_len_of(WIN), target)], dtype=torch.int32)
                prob = model((x, gm), y, (xl, gl), y_len)[0]
            probs.append(float(prob.reshape(-1)[0]))
        scores[name] = max(probs)
    return scores


def _chunk_order(name):
    try:
        return int(name.split("_")[1].split(".")[0])
    except (IndexError, ValueError):
        return name


def main():
    import librosa
    ap = argparse.ArgumentParser(description="PhonMatchNet keyword detector (local).")
    ap.add_argument("--keyword", required=True)
    ap.add_argument("--audio-dir", default=None, help="folder of .wav chunks")
    ap.add_argument("--wav", default=None, help="a single .wav")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--ground-truth", default=None,
                    help="comma-separated wav names that contain the keyword (for F1)")
    args = ap.parse_args()

    if args.wav:
        files = [args.wav]
    elif args.audio_dir:
        files = sorted(glob.glob(os.path.join(args.audio_dir, "*.wav")))
    else:
        ap.error("provide --wav or --audio-dir")
    if not files:
        ap.error("no .wav files found")

    chunks = {os.path.basename(f): librosa.load(f, sr=SR)[0] for f in files}
    print(f"Computing Google embeddings for {len(chunks)} chunk(s)...")
    gemb_data = compute_gembs(chunks)
    print("Loading PhonMatchNet + scoring...")
    model = load_model()
    scores = score_chunks(model, gemb_data, args.keyword)

    gt = set(args.ground_truth.split(",")) if args.ground_truth else None
    print(f"\nKeyword '{args.keyword}'  (threshold {args.threshold}):")
    tp = fp = fn = tn = 0
    for name in sorted(scores, key=_chunk_order):
        det = scores[name] >= args.threshold
        line = f"  {name}: score={scores[name]:.3f}  {'DETECTED' if det else '-'}"
        if gt is not None:
            truth = name in gt
            tp += det and truth
            fp += det and not truth
            fn += (not det) and truth
            tn += (not det) and not truth
            line += f"  (truth={truth})"
        print(line)
    if gt is not None:
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        print(f"  -> P={p:.2f} R={r:.2f} F1={f1:.2f}  (TP={tp} FP={fp} FN={fn} TN={tn})")


if __name__ == "__main__":
    main()
