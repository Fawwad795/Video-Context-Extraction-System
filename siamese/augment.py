"""Augmentation to make the matcher robust to messy broadcast audio.

Two waveform-level ops (background noise at a target SNR, pitch/time perturbation) and one
spectrogram-level op (SpecAugment time/frequency masking). These mirror the ideation's
"noise, accents, speed" robustness goals.
"""
import glob
import os
import random

import librosa
import torch
import torchaudio

from .audio import fix_length, load_wav
from .config import AUDIO, SC_NOISE_DIR

# ---- Background noise (lazy-loaded from Speech Commands' _background_noise_) ----
_NOISE_BANK: list[torch.Tensor] | None = None


def _load_noise_bank() -> list[torch.Tensor]:
    bank: list[torch.Tensor] = []
    for f in glob.glob(os.path.join(str(SC_NOISE_DIR), "*.wav")):
        w, sr = load_wav(f)  # already mono float32 (samples,)
        if sr != AUDIO.sample_rate:
            w = torchaudio.functional.resample(w, sr, AUDIO.sample_rate)
        bank.append(w)
    return bank


def add_background_noise(waveform: torch.Tensor, snr_db: float = 10.0) -> torch.Tensor:
    """Mix in a random slice of background noise scaled to the requested SNR (in dB)."""
    global _NOISE_BANK
    if _NOISE_BANK is None:
        _NOISE_BANK = _load_noise_bank()
    if not _NOISE_BANK:
        return fix_length(waveform)

    wav = fix_length(waveform)
    noise = random.choice(_NOISE_BANK)
    if noise.numel() < wav.numel():
        noise = torch.nn.functional.pad(noise, (0, wav.numel() - noise.numel()))
    start = random.randint(0, noise.numel() - wav.numel())
    noise = noise[start:start + wav.numel()]

    sig_power = wav.pow(2).mean()
    noise_power = noise.pow(2).mean() + 1e-10
    scale = torch.sqrt(sig_power / (noise_power * (10 ** (snr_db / 10))))
    return wav + scale * noise


# ---- SpecAugment (operates on a (1, n_mels, frames) spectrogram) ----
_freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=12)
_time_mask = torchaudio.transforms.TimeMasking(time_mask_param=20)


def spec_augment(spec: torch.Tensor, n_freq: int = 2, n_time: int = 2) -> torch.Tensor:
    out = spec.clone()
    for _ in range(n_freq):
        out = _freq_mask(out)
    for _ in range(n_time):
        out = _time_mask(out)
    return out


# ---- Waveform pitch / speed perturbation (via librosa) ----
def pitch_shift(waveform: torch.Tensor, n_steps: float) -> torch.Tensor:
    """Shift pitch by n_steps semitones, preserving length."""
    y = librosa.effects.pitch_shift(
        fix_length(waveform).numpy(), sr=AUDIO.sample_rate, n_steps=n_steps
    )
    return torch.from_numpy(y)


def time_stretch(waveform: torch.Tensor, rate: float) -> torch.Tensor:
    """Stretch/compress in time by `rate` (>1 faster), then re-fix to clip length."""
    y = librosa.effects.time_stretch(fix_length(waveform).numpy(), rate=rate)
    return fix_length(torch.from_numpy(y))
