"""Stage A (run in TF 'preprocess' docker): compute Google Speech embeddings
for sliding windows over the VMS chunks. Mirrors preprocess.py's padding.

Windows: 1.5s (24000 samples) @ 0.5s hop, padded to maxlen_a=32000 (2s) for
the embedder, matching the LibriPhrase training distribution (<=2s phrases).

Output: /out/vms_gemb.npz  with per-chunk  <name>__win [N,24000],
        <name>__gemb [N,G,96], <name>__start [N]
"""
import glob
import os

import librosa
import numpy as np
from speech_embedding import GoogleSpeechEmbedder

SR = 16000
WIN = 24000        # 1.5 s analysis window
HOP = 8000         # 0.5 s hop
MAXLEN_A = 32000   # pad to 2 s for the embedder (preprocess convention)
CHUNK_DIR = "/c"
OUT = "/out/vms_gemb.npz"


def main():
    emb = GoogleSpeechEmbedder()
    files = sorted(glob.glob(os.path.join(CHUNK_DIR, "*.wav")),
                   key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]))
    out = {}
    for f in files:
        name = os.path.basename(f)
        y, _ = librosa.load(f, sr=SR)               # resample 44.1k -> 16k
        starts = list(range(0, max(1, len(y) - WIN + 1), HOP))
        wins, gembs = [], []
        for s in starts:
            w = y[s:s + WIN].astype(np.float32)
            if len(w) < WIN:
                w = np.pad(w, (0, WIN - len(w)))
            wp = np.pad(w, (0, MAXLEN_A - WIN))      # pad to maxlen_a for embedder
            g = emb(wp[None, :]).numpy()[0]          # (G, 96)
            wins.append(w)
            gembs.append(g)
        out[name + "__win"] = np.stack(wins).astype(np.float32)
        out[name + "__gemb"] = np.stack(gembs).astype(np.float32)
        out[name + "__start"] = (np.array(starts) / SR).astype(np.float32)
        print(f"{name}: {len(starts)} windows, gemb {gembs[0].shape}", flush=True)

    np.savez(OUT, **out)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
