"""Siamese CNN encoder + contrastive loss (roadmap phase P2).

A shared VGG-style CNN maps a log-mel spectrogram to an L2-normalized embedding. Two clips
are compared by cosine similarity in that space, replacing the brittle np.correlate score.
Two sizes:
  - "full":    5 conv banks, 128-d embedding  (best accuracy; trained on the GPU box)
  - "reduced": 3 conv banks,  64-d embedding  (lightweight Jetson/edge target)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBank(nn.Module):
    """Two Conv-BN-ReLU layers + 2x2 max-pool (VGG-style)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class SiameseEncoder(nn.Module):
    """log-mel (B,1,n_mels,frames) -> L2-normalized embedding (B, embed_dim).

    Global average pooling makes the encoder agnostic to the exact spectrogram size, so the
    same weights work on variable-length clips at inference.
    """

    def __init__(self, channels=(32, 64, 128, 128, 256), embed_dim: int = 128):
        super().__init__()
        banks, in_ch = [], 1
        for c in channels:
            banks.append(ConvBank(in_ch, c))
            in_ch = c
        self.features = nn.Sequential(*banks)
        self.fc = nn.Linear(in_ch, embed_dim)
        self.embed_dim = embed_dim

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = h.mean(dim=(2, 3))          # global average pool -> (B, C)
        h = self.fc(h)
        return F.normalize(h, dim=1)    # unit-length embeddings

    def forward(self, x1, x2):
        return self.embed(x1), self.embed(x2)


# Named configurations referenced from the ideation (full vs reduced/edge).
CONFIGS = {
    "full": dict(channels=(32, 64, 128, 128, 256), embed_dim=128),
    "reduced": dict(channels=(32, 64, 64), embed_dim=64),
}


def build_encoder(kind: str = "full") -> SiameseEncoder:
    if kind not in CONFIGS:
        raise ValueError(f"model must be one of {list(CONFIGS)}, got {kind!r}")
    return SiameseEncoder(**CONFIGS[kind])


def contrastive_loss(e1, e2, y, margin: float = 1.0):
    """Hadsell-Chopra-LeCun contrastive loss on Euclidean distance of the embeddings.

    Same-word pairs (y=1) are pulled together; different-word pairs (y=0) are pushed apart
    until they are at least `margin` away. For unit-length embeddings d^2 = 2(1 - cos), so
    minimizing this directly improves the cosine score used at inference.
    """
    d = F.pairwise_distance(e1, e2)
    loss = y * d.pow(2) + (1.0 - y) * F.relu(margin - d).pow(2)
    return loss.mean()


def cosine_score(e1, e2):
    """Cosine similarity in [-1, 1]; for L2-normalized embeddings this is the dot product."""
    return (e1 * e2).sum(dim=1)
