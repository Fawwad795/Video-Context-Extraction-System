"""A/B evaluation of detector scoring variants on the Set-D chunks.

Implements steps 1-2 of the siamese/optimizations plan, no retraining:

  step 1a  WavLM mid-layer frame features (layer sweep) instead of the
           wav2vec2-base last layer + trained head;
  step 1b  soft-ZCA whitening fitted on the in-domain impostor cohort;
  step 1c  multi-template max-of-k and diagonal-Gaussian LLR scoring
           instead of single-centroid cosine;
  step 2   MaxSim late interaction: S(A, W) = mean_i max_j cos(a_i, w_j)
           over FRAME sequences (order-sensitive-ish, ColBERT-style).

Everything is evaluated as a *ranking* problem over the 10 Set-D chunks
per keyword (chunk score = max over sliding windows, same multi-scale
windowing as detector.py). Metrics per (keyword, config):

  AP      average precision of the chunk ranking vs transcript truth;
  margin  min(true chunk score) - max(false chunk score); > 0 means the
          keyword is perfectly separable by a threshold on this chunk set;
  bestF1  best chunk-level F1 over all thresholds.

The production detector/verifier pipeline is untouched; this is an
offline experiment harness. Ground truth = keyword present as a token in
audios/transcripts.txt (same rule as validate_detection.py).

Run:  python pipeline/eval_scoring_ab.py --keywords russia,weather,scotland,ireland
Writes a per-window/per-chunk score dump to logs/ab_scores.json.
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import librosa
import numpy as np

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "core"))

from scoring import (PROJECT_ROOT, SAMPLE_RATE, asnorm_windows, fit_llr_scorer,
                     fit_whitener, l2_normalize, list_chunk_audios,
                     max_template_scores, sample_stream_windows, whiten)
from embedders import (FRAME_STRIDE, BaselineHeadEmbedder, FrameBackend,
                       pool_segment, pooled_windows, samples_to_frames)

SCALES = (0.6, 0.8, 1.0)
MIN_CLIP_SECONDS = 0.15


# ---------------------------------------------------------------- ground truth

def load_truth(keyword):
    """chunk name -> bool, from audios/transcripts.txt token match."""
    path = os.path.join(PROJECT_ROOT, "audios", "transcripts.txt")
    truth, current = {}, None
    for line in open(path, encoding="utf-8"):
        m = re.match(r"\[(live_\d+\.wav)\]", line.strip())
        if m:
            current = m.group(1)
            truth.setdefault(current, False)
        elif current:
            tokens = set(re.findall(r"[a-z']+", line.lower()))
            if keyword.lower() in tokens:
                truth[current] = True
    return truth


# ---------------------------------------------------------------- data loading

def load_anchor_clips(keyword):
    """Trimmed kNN-VC converted clips (same guards as convert_anchor_knnvc)."""
    clip_dir = os.path.join(PROJECT_ROOT, "keywords", f"{keyword}_variants_knnvc")
    wavs = sorted(glob.glob(os.path.join(clip_dir, "*.wav")))
    if not wavs:
        raise FileNotFoundError(f"No converted clips in {clip_dir}")
    clips = []
    for w in wavs:
        y, _ = librosa.load(w, sr=SAMPLE_RATE)
        t, _ = librosa.effects.trim(y, top_db=30)
        if len(t) >= MIN_CLIP_SECONDS * SAMPLE_RATE:
            clips.append(t.astype(np.float32))
    median_len = np.median([len(c) for c in clips])
    clips = [c for c in clips if len(c) <= 1.8 * median_len]
    window_samples = int(np.median([len(c) for c in clips]))
    return clips, window_samples


# ---------------------------------------------------------------- metrics

def average_precision(labels, scores):
    order = np.argsort(-np.asarray(scores))
    labels = np.asarray(labels)[order]
    if labels.sum() == 0:
        return float("nan")
    hits, precisions = 0, []
    for i, l in enumerate(labels, start=1):
        if l:
            hits += 1
            precisions.append(hits / i)
    return float(np.mean(precisions))


def best_f1(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    best = 0.0
    for t in sorted(set(scores)):
        pred = np.asarray(scores) >= t
        tp = int((pred & labels).sum())
        fp = int((pred & ~labels).sum())
        fn = int((~pred & labels).sum())
        if tp:
            p, r = tp / (tp + fp), tp / (tp + fn)
            best = max(best, 2 * p * r / (p + r))
    return best


def margin(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    if labels.sum() == 0 or (~labels).sum() == 0:
        return float("nan")
    return float(scores[labels].min() - scores[~labels].max())


# ---------------------------------------------------------------- maxsim

def maxsim_chunk_scores(anchor_frames_list, chunk_frames, win_frames):
    """Best MaxSim score over sliding windows of a chunk, and its position.

    anchor_frames_list: list of [M_k, D] (L2-normalized) anchor templates.
    chunk_frames: [T, D] L2-normalized chunk frames.
    S(A, W) = mean_i max_{j in W} cos(a_i, c_j); chunk score = max over
    windows and templates. Sliding max via maximum_filter1d.
    """
    from scipy.ndimage import maximum_filter1d
    T = chunk_frames.shape[0]
    if T < win_frames:
        return -1.0, 0.0
    best_score, best_start = -1.0, 0
    for A in anchor_frames_list:
        sim = A @ chunk_frames.T                       # [M, T]
        # max over each length-win_frames window along T, evaluated at the
        # window's left edge: origin shift so index t covers [t, t+wf)
        wmax = maximum_filter1d(sim, size=win_frames, axis=1,
                                mode="constant", cval=-1.0,
                                origin=-(win_frames // 2))
        n_starts = T - win_frames + 1
        scores = wmax[:, :n_starts].mean(axis=0)       # [n_starts]
        i = int(np.argmax(scores))
        if scores[i] > best_score:
            best_score, best_start = float(scores[i]), i
    return best_score, best_start * FRAME_STRIDE / SAMPLE_RATE


# ---------------------------------------------------------------- evaluation

def eval_keyword(keyword, backends, layers, n_cohort, seed, maxsim_templates):
    truth = load_truth(keyword)
    clips, window_samples = load_anchor_clips(keyword)
    print(f"\n=== {keyword}: {len(clips)} anchor clips, "
          f"window {window_samples / SAMPLE_RATE:.2f}s, "
          f"truth: {sorted(n for n, t in truth.items() if t)}")

    rng = np.random.default_rng(seed)
    cohort_audio = sample_stream_windows(window_samples, n_cohort, rng)
    chunk_files = list_chunk_audios()
    chunk_names = [os.path.basename(f) for f in chunk_files]
    labels = [truth.get(n, False) for n in chunk_names]

    rows = []          # (config, chunk_scores dict)
    dump = defaultdict(dict)

    for backend_name, backend in backends.items():
        use_layers = layers if backend_name != "baseline-head" else [None]

        # ---- embed anchor clips + cohort once per layer
        anchor_by_layer = defaultdict(list)     # layer -> [K, D] pooled
        aframes_by_layer = defaultdict(list)    # layer -> list of [M, D]
        for c in clips:
            if backend_name == "baseline-head":
                fr = backend.frames(c)
                anchor_by_layer[None].append(pool_segment(fr))
            else:
                for l, fr in backend.frames(c, use_layers).items():
                    anchor_by_layer[l].append(pool_segment(fr))
                    aframes_by_layer[l].append(l2_normalize(fr))
        cohort_by_layer = defaultdict(list)
        for c in cohort_audio:
            if backend_name == "baseline-head":
                cohort_by_layer[None].append(pool_segment(backend.frames(c)))
            else:
                for l, fr in backend.frames(c, use_layers).items():
                    cohort_by_layer[l].append(pool_segment(fr))

        # ---- chunk frames once per layer
        chunk_frames = {}   # (chunk, layer) -> [T, D]
        for f, name in zip(chunk_files, chunk_names):
            y, _ = librosa.load(f, sr=SAMPLE_RATE)
            if backend_name == "baseline-head":
                chunk_frames[(name, None)] = backend.frames(y)
            else:
                for l, fr in backend.frames(y, use_layers).items():
                    chunk_frames[(name, l)] = fr

        for layer in use_layers:
            templates = np.stack(anchor_by_layer[layer])       # [K, D]
            cohort = np.stack(cohort_by_layer[layer])          # [N, D]
            if backend_name == "baseline-head":
                templates = backend.project(templates)
                cohort = backend.project(cohort)
            templates_n = l2_normalize(templates)
            centroid = l2_normalize(templates_n.mean(axis=0))
            cohort_n = l2_normalize(cohort)

            mu, W = fit_whitener(cohort)
            templates_w = whiten(templates, mu, W)
            centroid_w = l2_normalize(templates_w.mean(axis=0))
            cohort_w = whiten(cohort, mu, W)
            llr = fit_llr_scorer(whiten(templates, mu, W, renorm=False),
                                 whiten(cohort, mu, W, renorm=False))

            tag = (backend_name if layer is None
                   else f"{backend_name.split('/')[-1]}-L{layer}")
            configs = {
                f"{tag}|cos-centroid": None,
                f"{tag}|asnorm-centroid": None,
                f"{tag}|whiten-cos-centroid": None,
                f"{tag}|whiten-maxK": None,
                f"{tag}|whiten-llr": None,
            }
            per_cfg_scores = {c: [] for c in configs}

            for name in chunk_names:
                fr = chunk_frames[(name, layer)]
                scale_scores = defaultdict(lambda: -1e9)
                for scale in SCALES:
                    wf = samples_to_frames(
                        max(int(window_samples * scale),
                            int(0.15 * SAMPLE_RATE)))
                    pooled, _ = pooled_windows(fr, wf)
                    if len(pooled) == 0:
                        continue
                    if backend_name == "baseline-head":
                        pooled = backend.project(pooled)
                    pooled_n = l2_normalize(pooled)
                    pooled_w = whiten(pooled, mu, W)

                    s = pooled_n @ centroid
                    scale_scores[f"{tag}|cos-centroid"] = max(
                        scale_scores[f"{tag}|cos-centroid"], float(s.max()))
                    normed, _ = asnorm_windows(pooled_n, centroid, cohort_n)
                    scale_scores[f"{tag}|asnorm-centroid"] = max(
                        scale_scores[f"{tag}|asnorm-centroid"], float(normed.max()))
                    sw = pooled_w @ centroid_w
                    scale_scores[f"{tag}|whiten-cos-centroid"] = max(
                        scale_scores[f"{tag}|whiten-cos-centroid"], float(sw.max()))
                    smk = max_template_scores(pooled_w, templates_w)
                    scale_scores[f"{tag}|whiten-maxK"] = max(
                        scale_scores[f"{tag}|whiten-maxK"], float(smk.max()))
                    sllr = llr(whiten(pooled, mu, W, renorm=False))
                    scale_scores[f"{tag}|whiten-llr"] = max(
                        scale_scores[f"{tag}|whiten-llr"], float(sllr.max()))

                for cfg in configs:
                    per_cfg_scores[cfg].append(scale_scores[cfg])
                    dump[cfg][name] = scale_scores[cfg]

            for cfg, scores in per_cfg_scores.items():
                rows.append((cfg, scores))

            # ---- step 2: MaxSim on this layer's frames
            if layer is not None:
                aframes = aframes_by_layer[layer][:maxsim_templates]
                cfg = f"{tag}|maxsim"
                scores = []
                for name in chunk_names:
                    fr_n = l2_normalize(chunk_frames[(name, layer)])
                    wf = samples_to_frames(window_samples)
                    s, t = maxsim_chunk_scores(aframes, fr_n, wf)
                    scores.append(s)
                    dump[cfg][name] = s
                rows.append((cfg, scores))

    print(f"\n{'config':<44}{'AP':>7}{'margin':>9}{'bestF1':>8}")
    print("-" * 68)
    summary = []
    for cfg, scores in rows:
        ap = average_precision(labels, scores)
        mg = margin(labels, scores)
        f1 = best_f1(labels, scores)
        summary.append({"keyword": keyword, "config": cfg, "ap": ap,
                        "margin": mg, "best_f1": f1})
        print(f"{cfg:<44}{ap:>7.3f}{mg:>9.3f}{f1:>8.2f}")
    return summary, {cfg: dict(v) for cfg, v in dump.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--keywords", default="russia,weather,scotland,ireland")
    ap.add_argument("--backbone", default="microsoft/wavlm-base-plus")
    ap.add_argument("--layers", default="4,6,8,10,12")
    ap.add_argument("--n-cohort", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--maxsim-templates", type=int, default=8)
    ap.add_argument("--skip-baseline", action="store_true")
    args = ap.parse_args()

    layers = [int(l) for l in args.layers.split(",")]
    backends = {}
    if not args.skip_baseline:
        weights = os.environ.get(
            "SIAMESE_WEIGHTS",
            os.path.join(PROJECT_ROOT, "checkpoints", "best_siamese_model.pth"))
        backends["baseline-head"] = BaselineHeadEmbedder(weights)
    backends[args.backbone] = FrameBackend(args.backbone)

    all_summary, all_dump = [], {}
    for kw in args.keywords.split(","):
        kw = kw.strip()
        summary, dump = eval_keyword(kw, backends, layers, args.n_cohort,
                                     args.seed, args.maxsim_templates)
        all_summary.extend(summary)
        all_dump[kw] = dump

    # macro summary over keywords per config
    print(f"\n{'=== MACRO (mean over keywords) ===':<44}")
    by_cfg = defaultdict(list)
    for row in all_summary:
        by_cfg[row["config"]].append(row)
    print(f"{'config':<44}{'AP':>7}{'margin':>9}{'bestF1':>8}")
    print("-" * 68)
    ranked = sorted(by_cfg.items(),
                    key=lambda kv: -np.nanmean([r["ap"] for r in kv[1]]))
    for cfg, rws in ranked:
        print(f"{cfg:<44}"
              f"{np.nanmean([r['ap'] for r in rws]):>7.3f}"
              f"{np.nanmean([r['margin'] for r in rws]):>9.3f}"
              f"{np.nanmean([r['best_f1'] for r in rws]):>8.2f}")

    out = os.path.join(PROJECT_ROOT, "logs", "ab_scores.json")
    with open(out, "w") as f:
        json.dump({"summary": all_summary, "chunk_scores": all_dump}, f, indent=2)
    print(f"\nScore dump: {out}")


if __name__ == "__main__":
    main()
