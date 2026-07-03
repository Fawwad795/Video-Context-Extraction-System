"""Embedding backends for the scoring A/B experiments (siamese/optimizations).

Two families:

  FrameBackend    - a frozen SSL backbone (WavLM / wav2vec2) run ONCE over a
                    full waveform with output_hidden_states=True. Window
                    embeddings for every layer are then derived by mean-pooling
                    frame features inside the window (cumsum trick), so a
                    whole layer sweep costs a single forward pass per chunk.
                    This follows the acoustic-word-embedding literature
                    (Sanabria et al. 2023; Pasad et al. 2021): word identity
                    peaks in MIDDLE transformer layers of pretrained models,
                    not the last layer.

  BaselineHeadEmbedder - reproduces the production Siamese embedding
                    (wav2vec2-base last layer, mean pooled, projection head)
                    under the same full-chunk protocol, for a fair same-
                    harness baseline.

Conv frontend of both models: 20 ms stride, 25 ms receptive field, so
n_frames = floor((n_samples - 400) / 320) + 1 and frame t covers
[320*t, 320*t + 400) samples.
"""

import numpy as np
import torch
import torch.nn as nn

SAMPLE_RATE = 16000
FRAME_STRIDE = 320   # samples per output frame (20 ms)
FRAME_RECEPTIVE = 400

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def n_frames_for(n_samples):
    return max(0, (n_samples - FRAME_RECEPTIVE) // FRAME_STRIDE + 1)


def samples_to_frames(n_samples):
    """Window length in frames for a window of n_samples samples."""
    return max(1, int(round(n_samples / FRAME_STRIDE)))


class FrameBackend:
    """Frozen SSL backbone -> per-layer frame features for a waveform."""

    def __init__(self, model_name="microsoft/wavlm-base-plus"):
        from transformers import AutoFeatureExtractor, AutoModel
        print(f"Loading frame backend: {model_name}")
        self.name = model_name
        self.fe = AutoFeatureExtractor.from_pretrained(model_name)
        try:
            self.model = AutoModel.from_pretrained(model_name)
        except (ValueError, OSError):
            # transformers blocks torch.load .bin checkpoints on torch < 2.6
            # (CVE-2025-32434); fall back to a safetensors copy (the GPU box
            # carries one converted into its HF cache).
            self.model = AutoModel.from_pretrained(model_name, use_safetensors=True)
        self.model.eval().to(_DEVICE)
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def frames(self, audio, layers):
        """audio: float32 [T] at 16 kHz -> {layer: [n_frames, D] float32}.

        layer indexing: 0 = conv output, 1..12 = transformer layers.
        """
        inputs = self.fe(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}
        out = self.model(**inputs, output_hidden_states=True)
        hs = out.hidden_states  # tuple of [1, T', D], len = n_layers + 1
        return {l: hs[l][0].cpu().numpy().astype(np.float32) for l in layers}


def pooled_windows(frames, win_frames, hop_frames=1):
    """Mean-pool sliding windows over frame features via cumsum.

    frames: [T, D] -> (embs [N, D], start_frame_indices [N])
    """
    T = frames.shape[0]
    if T < win_frames:
        return np.empty((0, frames.shape[1]), np.float32), np.empty(0, np.int64)
    csum = np.concatenate([np.zeros((1, frames.shape[1]), frames.dtype),
                           np.cumsum(frames, axis=0)], axis=0)
    starts = np.arange(0, T - win_frames + 1, hop_frames)
    embs = (csum[starts + win_frames] - csum[starts]) / float(win_frames)
    return embs.astype(np.float32), starts


def pool_segment(frames):
    """Mean-pool a whole clip's frames -> [D]."""
    return frames.mean(axis=0).astype(np.float32)


class WavLMEmbedder:
    """Drop-in replacement for SiameseAudioModel: frozen SSL mid layer.

    Exposes the same get_embedding(batch, sr) interface that
    scoring.embed_batch() expects, plus a frames() fast path so the
    detector can embed all sliding windows of a chunk from a single
    forward pass (mean-pooled frame features, cumsum trick).
    """

    is_frame_backend = True

    def __init__(self, model_name="microsoft/wavlm-base-plus", layer=10):
        self.backend = FrameBackend(model_name)
        self.layer = int(layer)
        print(f"Embedding backend: {model_name} layer {self.layer} (mean-pooled)")

    @torch.no_grad()
    def get_embedding(self, audio_list, sampling_rate=SAMPLE_RATE):
        """List of same-length float32 arrays -> [B, D] torch tensor."""
        inputs = self.backend.fe(audio_list, sampling_rate=sampling_rate,
                                 return_tensors="pt", padding=True)
        inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}
        out = self.backend.model(**inputs, output_hidden_states=True)
        return out.hidden_states[self.layer].mean(dim=1)

    def frames(self, audio):
        """Full-waveform frame features [T, D] at this backend's layer."""
        return self.backend.frames(audio, [self.layer])[self.layer]

    # no-op compatibility with load_siamese_model() call sites
    def eval(self):
        return self

    def to(self, device):
        return self


class AttentivePoolingHead(nn.Module):
    """Trainable pooling + projection over frozen SSL frames (Step 3).

    Multi-head attentive pooling (ICASSP 2021 QbE-style): each head scores
    every frame with a small tanh bottleneck, softmax-weights the frames,
    and the head-pooled vectors are averaged back to D. A residual MLP
    then reshapes the pooled embedding.

    Identity initialization: the score layer and the residual MLP's output
    layer are zero-initialized, so an UNTRAINED head is exactly uniform
    (mean) pooling plus an identity projection - i.e. it reproduces the
    validated wavlm-L10 mean-pool embedding bit-for-bit and training can
    only move away from that operating point.
    """

    def __init__(self, dim=768, n_heads=4, attn_hidden=128, proj_hidden=1024):
        super().__init__()
        self.config = {"dim": dim, "n_heads": n_heads,
                       "attn_hidden": attn_hidden, "proj_hidden": proj_hidden}
        self.pre = nn.Linear(dim, attn_hidden)
        self.score = nn.Linear(attn_hidden, n_heads)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)
        self.proj = nn.Sequential(
            nn.Linear(dim, proj_hidden), nn.GELU(),
            nn.Linear(proj_hidden, dim))
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, frames, mask=None):
        """frames [B, T, D], mask [B, T] bool (True = real frame) -> [B, D]."""
        logits = self.score(torch.tanh(self.pre(frames)))       # [B, T, H]
        if mask is not None:
            logits = logits.masked_fill(~mask.unsqueeze(-1), -1e4)
        w = logits.softmax(dim=1)
        pooled = torch.einsum("bth,btd->bhd", w, frames).mean(dim=1)
        return pooled + self.proj(pooled)


def save_head_checkpoint(head, path, extra=None):
    payload = {"head": head.state_dict(), "config": head.config}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_head_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    head = AttentivePoolingHead(**ckpt["config"])
    head.load_state_dict(ckpt["head"])
    head.eval()
    return head, ckpt


class TrainedWavLMEmbedder:
    """SIAMESE_BACKEND=wavlm-trained: frozen WavLM mid layer + trained head.

    Same interfaces as WavLMEmbedder (get_embedding / frames), plus
    pool_windows() so the detector derives all sliding-window embeddings
    of a chunk from ONE backbone forward pass - but pooled by the trained
    attentive head instead of the cumsum mean.
    """

    is_frame_backend = True

    def __init__(self, weights_path=None, model_name="microsoft/wavlm-base-plus",
                 layer=10):
        self.backend = FrameBackend(model_name)
        if weights_path:
            self.head, ckpt = load_head_checkpoint(weights_path)
            self.layer = int(ckpt.get("layer", layer))
            print(f"Trained attentive head loaded from {weights_path} "
                  f"(layer {self.layer})")
        else:
            self.head = AttentivePoolingHead()
            self.layer = int(layer)
            print("WARNING: no v3 checkpoint - identity-init head "
                  "(equivalent to mean-pooled wavlm backend).")
        self.head.eval().to(_DEVICE)
        for p in self.head.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def get_embedding(self, audio_list, sampling_rate=SAMPLE_RATE):
        inputs = self.backend.fe(audio_list, sampling_rate=sampling_rate,
                                 return_tensors="pt", padding=True)
        inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}
        out = self.backend.model(**inputs, output_hidden_states=True)
        return self.head(out.hidden_states[self.layer])

    def frames(self, audio):
        return self.backend.frames(audio, [self.layer])[self.layer]

    @torch.no_grad()
    def pool_windows(self, frames, win_frames, hop_frames=1, batch_size=256):
        """Head-pooled sliding windows over [T, D] frame features.

        Returns (embs [N, D] float32, start_frame_indices [N]) - same
        contract as pooled_windows().
        """
        T = frames.shape[0]
        if T < win_frames:
            return (np.empty((0, frames.shape[1]), np.float32),
                    np.empty(0, np.int64))
        fr = torch.from_numpy(np.ascontiguousarray(frames)).to(_DEVICE)
        windows = fr.unfold(0, win_frames, hop_frames)   # [N, D, win]
        windows = windows.permute(0, 2, 1)               # [N, win, D]
        out = []
        for i in range(0, windows.shape[0], batch_size):
            out.append(self.head(windows[i:i + batch_size]).cpu())
        embs = torch.cat(out).numpy().astype(np.float32)
        starts = np.arange(0, T - win_frames + 1, hop_frames, dtype=np.int64)
        return embs, starts

    # no-op compatibility with load_siamese_model() call sites
    def eval(self):
        return self

    def to(self, device):
        return self


class BaselineHeadEmbedder:
    """Production Siamese embedding under the frame-pooling protocol.

    wav2vec2-base LAST transformer layer, mean pooled per window, then the
    trained 768->256->128 projection head from the production checkpoint.
    """

    LAYER = 12

    def __init__(self, weights_path):
        self.backend = FrameBackend("facebook/wav2vec2-base")
        self.head = torch.nn.Sequential(
            torch.nn.Linear(768, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 128))
        state = torch.load(weights_path, map_location="cpu")
        self.head.load_state_dict(state)
        self.head.eval()
        print(f"Baseline projection head loaded from {weights_path}")

    def frames(self, audio):
        return self.backend.frames(audio, [self.LAYER])[self.LAYER]

    @torch.no_grad()
    def project(self, pooled):
        """pooled: [N, 768] mean-pooled features -> [N, 128] head output."""
        pooled = np.atleast_2d(pooled)
        out = self.head(torch.from_numpy(pooled.astype(np.float32)))
        return out.numpy().astype(np.float32)
