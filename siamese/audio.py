"""Audio front-end: load a mono waveform and turn it into a fixed-size log-mel spectrogram.

This is the shared input representation for both legs of the Siamese network. Keeping it
in one place guarantees the live-stream chunk and the synthesized keyword are featurized
identically at inference time.
"""
import soundfile as sf
import torch
import torchaudio

from .config import AUDIO

# A single shared transform instance (stateless, safe to reuse across calls).
_mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=AUDIO.sample_rate,
    n_fft=AUDIO.n_fft,
    win_length=AUDIO.win_length,
    hop_length=AUDIO.hop_length,
    n_mels=AUDIO.n_mels,
    f_min=AUDIO.f_min,
    f_max=AUDIO.f_max,
    power=2.0,
)
_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=AUDIO.top_db)


def load_wav(path: str) -> tuple[torch.Tensor, int]:
    """Load a WAV as a mono float32 tensor of shape (samples,) plus its sample rate.

    Uses soundfile directly: torchaudio 2.11 removed its soundfile/sox backends and routes
    ``torchaudio.load`` through torchcodec, which we avoid for portability on Windows.
    """
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    wav = torch.from_numpy(data)
    if wav.dim() > 1:                 # (samples, channels) -> mono
        wav = wav.mean(dim=1)
    return wav, sr


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Collapse any channel dimension to a single mono track of shape (samples,)."""
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=0)
    return waveform


def fix_length(waveform: torch.Tensor, num_samples: int | None = None) -> torch.Tensor:
    """Zero-pad (right) or center-trim a mono waveform to a fixed number of samples."""
    num_samples = num_samples or AUDIO.num_samples
    waveform = to_mono(waveform)
    n = waveform.numel()
    if n < num_samples:
        waveform = torch.nn.functional.pad(waveform, (0, num_samples - n))
    elif n > num_samples:
        start = (n - num_samples) // 2
        waveform = waveform[start:start + num_samples]
    return waveform


def log_mel(waveform: torch.Tensor) -> torch.Tensor:
    """Mono waveform -> (1, n_mels, num_frames) per-instance-normalized log-mel spectrogram."""
    wav = fix_length(waveform)
    spec = _to_db(_mel(wav))                      # (n_mels, frames), in dB
    spec = (spec - spec.mean()) / (spec.std() + 1e-5)   # per-instance standardization
    return spec.unsqueeze(0)                      # (1, n_mels, frames) — add channel dim
