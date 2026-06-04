"""Real-time Siamese keyword detector for VMS integration (roadmap phase P4).

Loads a trained Siamese encoder once, builds a keyword PROTOTYPE embedding from the VMS's
existing synthesized keyword clips (the 7-accent HiFi-GAN output in Stream*_searchword1), and
scores live audio chunks by cosine similarity against that prototype.

This is the drop-in replacement for the np.correlate "matching percentage" in
Research/Stream*_corelation_updated_v2.py:
    score = detector.score_chunk(chunk_wav)        # in [-1, 1]
    if score >= detector.threshold: ...            # calibrated, not a hand-tuned 70%
"""
from __future__ import annotations

import glob
import os

import torch
import torch.nn.functional as F
import torchaudio

from .audio import AUDIO, fix_length, load_wav, log_mel, to_mono
from .model import build_encoder


class SiameseDetector:
    def __init__(self, ckpt_path: str, device: str = "cpu",
                 threshold: float = 0.75, window_hop_s: float = 0.5):
        self.device = torch.device(device)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model = build_encoder(ckpt["model"]).to(self.device).eval()
        self.model.load_state_dict(ckpt["state_dict"])
        self.model_kind = ckpt["model"]
        self.threshold = threshold
        self.window_hop_s = window_hop_s
        self.prototype: torch.Tensor | None = None
        self.keyword: str | None = None

    @torch.no_grad()
    def _embed_specs(self, specs: list[torch.Tensor]) -> torch.Tensor:
        """Embed a list of (1, n_mels, frames) specs -> (N, d) unit-norm embeddings."""
        batch = torch.stack(specs).to(self.device)        # (N, 1, n_mels, frames)
        return self.model.embed(batch).cpu()

    @torch.no_grad()
    def build_prototype(self, keyword_dir: str, keyword: str | None = None):
        """Average embeddings of the synthesized keyword clips into one unit prototype vector.

        `keyword_dir` is the VMS Stream*_searchword1 folder; files are named
        '{speaker}-{word}.wav'. If `keyword` is given, only that word's clips are used.
        """
        paths = sorted(glob.glob(os.path.join(keyword_dir, "*.wav")))
        if keyword is not None:
            paths = [p for p in paths
                     if os.path.basename(p).split("-", 1)[-1].rsplit(".", 1)[0] == keyword]
        if not paths:
            raise FileNotFoundError(f"No keyword clips in {keyword_dir} (keyword={keyword})")

        specs = []
        for p in paths:
            wav, sr = load_wav(p)
            if sr != AUDIO.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, AUDIO.sample_rate)
            specs.append(log_mel(fix_length(wav)))
        embs = self._embed_specs(specs)                   # (n_accents, d)
        self.prototype = F.normalize(embs.mean(0), dim=0)
        self.keyword = keyword
        return self.prototype, len(paths)

    @torch.no_grad()
    def score_chunk(self, wav: torch.Tensor) -> float:
        """Max cosine similarity between the prototype and any 1-second window of the chunk.

        A live chunk is ~5 s but the model is trained on ~1 s words, so we slide a 1 s window
        across the chunk and take the best match (the keyword may appear anywhere in it).
        """
        if self.prototype is None:
            raise RuntimeError("call build_prototype() before scoring")
        wav = to_mono(wav)
        win = AUDIO.num_samples
        hop = max(1, int(self.window_hop_s * AUDIO.sample_rate))
        if wav.numel() <= win:
            windows = [fix_length(wav)]
        else:
            windows = [wav[s:s + win] for s in range(0, wav.numel() - win + 1, hop)]
            if (wav.numel() - win) % hop != 0:
                windows.append(wav[-win:])                # include the tail
        embs = self._embed_specs([log_mel(w) for w in windows])   # (W, d)
        sims = embs @ self.prototype                              # (W,)
        return float(sims.max())

    def score_file(self, wav_path: str) -> float:
        wav, sr = load_wav(wav_path)
        if sr != AUDIO.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, AUDIO.sample_rate)
        return self.score_chunk(wav)

    def is_match(self, wav_path: str) -> bool:
        return self.score_file(wav_path) >= self.threshold
