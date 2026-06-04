"""Central configuration for the Siamese keyword-spotting extension.

All audio + pipeline hyperparameters live here so the data pipeline, training,
evaluation, and the eventual VMS integration share one source of truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---- Repo-relative paths --------------------------------------------------
SIAMESE_DIR = Path(__file__).resolve().parent
DATA_DIR = SIAMESE_DIR / "data"
# torchaudio extracts the corpus to DATA_DIR/SpeechCommands/speech_commands_v0.02
SC_ROOT = DATA_DIR
SC_CORPUS = DATA_DIR / "SpeechCommands" / "speech_commands_v0.02"
SC_NOISE_DIR = SC_CORPUS / "_background_noise_"
TTS_CACHE_DIR = DATA_DIR / "tts_cache"
ARTIFACTS_DIR = SIAMESE_DIR / "artifacts"
CHECKPOINT_DIR = SIAMESE_DIR / "checkpoints"

# torchaudio's reserved folder of long noise recordings (not a real keyword class)
NOISE_LABEL = "_background_noise_"


@dataclass(frozen=True)
class AudioConfig:
    """Log-mel front-end parameters. FBank/log-mel chosen over MFCC per Li & Song (2021)."""
    sample_rate: int = 16_000
    clip_seconds: float = 1.0       # Speech Commands clips are ~1 s
    n_mels: int = 64
    n_fft: int = 400                # 25 ms @ 16 kHz
    win_length: int = 400
    hop_length: int = 160           # 10 ms @ 16 kHz
    f_min: float = 20.0
    f_max: float = 8_000.0
    top_db: float = 80.0

    @property
    def num_samples(self) -> int:
        return int(self.sample_rate * self.clip_seconds)

    @property
    def num_frames(self) -> int:
        # torchaudio MelSpectrogram uses center=True -> 1 + n_samples // hop frames
        return self.num_samples // self.hop_length + 1


@dataclass(frozen=True)
class PairConfig:
    """How same/different-word pairs are sampled for the Siamese network."""
    neg_per_pos: int = 1            # 1:1 for quick checks; raise to 2 for training (per ideation)
    seed: int = 1
    max_pairs: int | None = None    # cap pair count for fast runs; None = use all


AUDIO = AudioConfig()
PAIRS = PairConfig()
