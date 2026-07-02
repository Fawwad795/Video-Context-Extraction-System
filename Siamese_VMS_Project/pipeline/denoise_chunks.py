"""Denoise captured chunks with spectral-gating noise reduction.

Experiment harness for "process the captured audio first, then match".
Reads <src>/*.wav, applies non-stationary spectral gating (noisereduce),
writes the cleaned chunks to <dst>/ with identical filenames so the rest
of the pipeline can score them via SIAMESE_AUDIO_DIR=<dst>.

Reports per-chunk RMS before/after as a sanity check that something
actually changed.
"""

import argparse
import glob
import os

import librosa
import numpy as np
import noisereduce as nr
import soundfile as sf

SR = 16000


def main():
    ap = argparse.ArgumentParser(description="Denoise chunks for the matching experiment.")
    ap.add_argument("--src", default="audios")
    ap.add_argument("--dst", default="audios_denoised")
    ap.add_argument("--stationary", action="store_true",
                    help="stationary noise profile (default: non-stationary)")
    ap.add_argument("--prop-decrease", type=float, default=1.0,
                    help="strength of reduction, 0..1")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.src, "*.wav")),
                   key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]))
    if not files:
        print(f"No .wav files in {args.src}")
        return

    print(f"Denoising {len(files)} chunks ({args.src} -> {args.dst}), "
          f"{'stationary' if args.stationary else 'non-stationary'} "
          f"prop_decrease={args.prop_decrease}")
    for f in files:
        name = os.path.basename(f)
        y, _ = librosa.load(f, sr=SR)
        reduced = nr.reduce_noise(y=y, sr=SR, stationary=args.stationary,
                                  prop_decrease=args.prop_decrease)
        sf.write(os.path.join(args.dst, name), reduced.astype(np.float32), SR)
        rms_in = float(np.sqrt(np.mean(y ** 2)))
        rms_out = float(np.sqrt(np.mean(reduced ** 2)))
        print(f"  {name}: RMS {rms_in:.4f} -> {rms_out:.4f} "
              f"({100 * (rms_out - rms_in) / max(rms_in, 1e-9):+.1f}%)")
    print(f"Done. Cleaned chunks in {args.dst}/")


if __name__ == "__main__":
    main()
