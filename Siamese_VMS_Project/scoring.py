"""Shared utilities for the Phase-1 adaptive scoring pipeline.

Embeddings are L2-normalized and compared with cosine similarity. Raw
cosine scores are then normalized with Adaptive S-norm (AS-norm) against
a cohort of impostor embeddings, so the detection threshold lives in
calibrated "standard deviations above the impostor distribution" units
instead of raw embedding-space distance:

    s_norm = 0.5 * ( (s - mu_a) / sd_a  +  (s - mu_w) / sd_w )

where (mu_a, sd_a) are the mean/std of the anchor's top-k cohort scores
and (mu_w, sd_w) the same for the test window. Top-k selection makes the
normalization adapt per-trial to the closest impostors (AS-norm), which
compensates for accent, speaking style, and room acoustics shifts.
"""

import glob
import os

import librosa
import numpy as np
import torch

from siamese_model import SiameseAudioModel

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# Override with the SIAMESE_WEIGHTS env var to A/B different checkpoints
WEIGHTS_PATH = os.environ.get(
    "SIAMESE_WEIGHTS", os.path.join(PROJECT_ROOT, "best_siamese_model.pth"))
# Override with SIAMESE_AUDIO_DIR to score against preprocessed audio
AUDIO_DIR = os.environ.get("SIAMESE_AUDIO_DIR", os.path.join(PROJECT_ROOT, "audios"))
SAMPLE_RATE = 16000
DEFAULT_TOP_K = 50


def cohort_path(keyword):
    # Per-keyword: cohort windows are sampled at the keyword's duration and
    # distractor words exclude the keyword, so cohorts are not interchangeable.
    return os.path.join(PROJECT_ROOT, f"cohort_{keyword}.npz")


def load_siamese_model():
    model = SiameseAudioModel()
    if os.path.exists(WEIGHTS_PATH):
        print(f"Loading checkpoint: {WEIGHTS_PATH}")
        model.load_weights(WEIGHTS_PATH)
    else:
        print(f"WARNING: {WEIGHTS_PATH} not found - using an untrained projection head.")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model


def l2_normalize(x, axis=-1):
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def embed_batch(model, audios, batch_size=16):
    """Embed a list of float32 audio arrays (16 kHz) -> L2-normalized [N, D].

    Arrays are grouped by length so the HF feature extractor never pads a
    batch; padding zeros would bias the mean-pooled embeddings.
    """
    by_len = {}
    for i, a in enumerate(audios):
        by_len.setdefault(len(a), []).append(i)

    out = [None] * len(audios)
    with torch.no_grad():
        for idxs in by_len.values():
            for start in range(0, len(idxs), batch_size):
                chunk = idxs[start:start + batch_size]
                batch = [np.asarray(audios[i], dtype=np.float32) for i in chunk]
                emb = model.get_embedding(batch, SAMPLE_RATE).cpu().numpy()
                for j, i in enumerate(chunk):
                    out[i] = emb[j]
    return l2_normalize(np.stack(out))


def topk_stats(scores, top_k):
    k = min(top_k, len(scores))
    top = np.partition(scores, -k)[-k:]
    return float(np.mean(top)), float(max(np.std(top), 1e-3))


def asnorm_windows(window_embs, anchor, cohort, top_k=DEFAULT_TOP_K):
    """AS-norm scores of [B, D] window embeddings against a [D] anchor.

    Returns (normalized_scores [B], raw_cosine_scores [B]).
    """
    window_embs = np.atleast_2d(window_embs)
    raw = window_embs @ anchor

    mu_a, sd_a = topk_stats(cohort @ anchor, top_k)

    k = min(top_k, cohort.shape[0])
    w_scores = window_embs @ cohort.T               # [B, N]
    top = np.partition(w_scores, -k, axis=1)[:, -k:]
    mu_w = top.mean(axis=1)
    sd_w = np.maximum(top.std(axis=1), 1e-3)

    normed = 0.5 * ((raw - mu_a) / sd_a + (raw - mu_w) / sd_w)
    return normed, raw


def load_cohort(keyword):
    path = cohort_path(keyword)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found - run cohort_builder.py --keyword {keyword} first.")
    data = np.load(path)
    return l2_normalize(data["embeddings"])


def list_chunk_audios(audio_dir=AUDIO_DIR):
    files = glob.glob(os.path.join(audio_dir, "*.wav"))
    files.sort(key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]))
    return files


def sample_stream_windows(window_samples, n_windows, rng, audio_dir=AUDIO_DIR):
    """Random keyword-length windows from the live chunks.

    These are (with overwhelming probability) non-keyword audio in exactly
    the deployment domain, which is what an impostor cohort should be.
    """
    files = list_chunk_audios(audio_dir)
    if not files:
        raise FileNotFoundError(
            f"No .wav chunks in {audio_dir} - run downloader.py first.")
    audios = []
    for f in files:
        y, _ = librosa.load(f, sr=SAMPLE_RATE)
        if len(y) > window_samples:
            audios.append(y)
    if not audios:
        raise ValueError("All chunks are shorter than the keyword window.")
    windows = []
    for _ in range(n_windows):
        y = audios[int(rng.integers(len(audios)))]
        start = int(rng.integers(0, len(y) - window_samples))
        windows.append(y[start:start + window_samples].astype(np.float32))
    return windows
