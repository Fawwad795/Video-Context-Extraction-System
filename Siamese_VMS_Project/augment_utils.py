"""Signal-level augmentation to "humanize" synthetic TTS audio.

Shared by keyword_generator.py (anchor building, PC) and dataset_v2.py
(Phase-2 training, AWS). Pure numpy/librosa/scipy - no project imports.
"""

import librosa
import numpy as np
import scipy.signal


def synthetic_rir(rng, rt60, sr=16000):
    """Exponentially-decaying noise impulse response (-60 dB at rt60)."""
    n = int(rt60 * sr)
    t = np.arange(n) / sr
    rir = rng.standard_normal(n).astype(np.float32) * np.exp(-6.9078 * t / rt60)
    rir[0] = 1.0
    return rir / np.max(np.abs(rir))


def augment_audio(y, rng, sr=16000):
    """One randomly 'humanized' copy: pitch / tempo / reverb / band-limit / noise."""
    out = y.astype(np.float32).copy()
    if rng.random() < 0.5:
        out = librosa.effects.pitch_shift(
            out, sr=sr, n_steps=float(rng.uniform(-1.5, 1.5)))
    if rng.random() < 0.5:
        out = librosa.effects.time_stretch(out, rate=float(rng.uniform(0.88, 1.12)))
    if rng.random() < 0.5:
        rir = synthetic_rir(rng, rt60=float(rng.uniform(0.1, 0.4)), sr=sr)
        wet = scipy.signal.fftconvolve(out, rir)[:len(out)].astype(np.float32)
        mix = float(rng.uniform(0.2, 0.5))
        out = (1.0 - mix) * out + mix * wet
    if rng.random() < 0.3:
        sos = scipy.signal.butter(
            2, float(rng.uniform(2500, 6000)), btype="low", fs=sr, output="sos")
        out = scipy.signal.sosfilt(sos, out).astype(np.float32)
    snr_db = float(rng.uniform(5, 25))
    sig_power = float(np.mean(out ** 2)) + 1e-12
    noise = rng.standard_normal(len(out)).astype(np.float32)
    noise *= np.sqrt(sig_power / (10 ** (snr_db / 10)) / (float(np.mean(noise ** 2)) + 1e-12))
    out = out + noise
    return (out / (np.max(np.abs(out)) + 1e-9) * 0.95).astype(np.float32)
