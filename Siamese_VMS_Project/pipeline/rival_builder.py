"""Build rival anchors for decision-time verification (core/rival_verify.py).

For the enrolled keyword: derive its phonetic near-neighbour words from text
(CMUdict/g2p_en edit distance), synthesize each rival in the 7 canonical
SpeechT5 voices, embed the clips with the active backend, and store one
centroid per rival plus the calibrated accept margin delta.

Usage: python pipeline/rival_builder.py --keyword party
Artifacts: keywords/<kw>_rivals<suffix>.npz, keywords/<kw>_rivals/<w>_<v>.wav
"""

import argparse
import os
import time
import warnings

import numpy as np

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "core"))

import console as ui
import rival_verify as rv
from scoring import (PROJECT_ROOT, SAMPLE_RATE, anchor_path, embed_batch,
                     l2_normalize, load_cohort, load_siamese_model)

warnings.filterwarnings("ignore")

import librosa
import soundfile as sf
import torch  # noqa: F401  (keyword_generator.synthesize expects it loaded)

from keyword_generator import CANONICAL_SPEAKERS, load_tts, synthesize


def synthesize_rivals(rivals, out_root):
    """Render every rival x 7 canonical voices; cached on disk."""
    missing = [(w, v) for w in rivals for v in CANONICAL_SPEAKERS
               if not os.path.exists(os.path.join(out_root, f"{w}_{v}.wav"))]
    if missing:
        processor, tts_model, vocoder, xvectors = load_tts()
        os.makedirs(out_root, exist_ok=True)
        for word, voice in missing:
            import torch as _t
            xv = _t.tensor(xvectors[CANONICAL_SPEAKERS[voice]]["xvector"])
            wav = synthesize(processor, tts_model, vocoder, word, xv)
            sf.write(os.path.join(out_root, f"{word}_{voice}.wav"),
                     wav, SAMPLE_RATE)
    clips = {}
    for w in rivals:
        clips[w] = []
        for v in CANONICAL_SPEAKERS:
            y, _ = librosa.load(os.path.join(out_root, f"{w}_{v}.wav"),
                                sr=SAMPLE_RATE)
            y, _ = librosa.effects.trim(y, top_db=30)
            if len(y) >= int(0.15 * SAMPLE_RATE):
                clips[w].append(y)
    return clips


def main():
    ap = argparse.ArgumentParser(description="Build rival anchors (RAV).")
    ap.add_argument("--keyword", required=True)
    ap.add_argument("--top-n", type=int, default=rv.N_RIVALS)
    args = ap.parse_args()
    keyword = args.keyword.lower().strip()

    t0 = time.perf_counter()
    ui.banner("RIVAL ANCHORS", f"keyword: {keyword}")

    npz_path = rv.rivals_path(keyword, PROJECT_ROOT)
    if os.path.exists(npz_path):
        words, _, delta = rv.load_rivals(npz_path)
        ui.ok(f"already built ({len(words)} rivals, delta {delta:+.3f}) - "
              f"delete {os.path.relpath(npz_path, PROJECT_ROOT)} to rebuild")
        return

    apath = anchor_path(keyword)
    if not os.path.exists(apath):
        ui.fail(f"{apath} not found - run keyword_generator.py first.")
        return
    z = np.load(apath)
    anchor, positives = z["centroid"], z["positives"]

    ui.step("selecting rival words (lexicon + embedding bank) ...")
    rivals = rv.find_rivals(keyword, anchor=anchor, top_n=args.top_n,
                            log=ui.item)
    if not rivals:
        ui.fail("no rivals found (lexicon miss?) - RAV unavailable.")
        return

    ui.step("synthesizing rival clips (7 canonical voices each) ...")
    clips = synthesize_rivals(
        rivals, os.path.join(PROJECT_ROOT, "keywords", f"{keyword}_rivals"))

    ui.step("embedding rival clips ...")
    model = load_siamese_model()
    try:
        cohort = load_cohort(keyword)
    except Exception:
        cohort = None
        ui.warn("no cohort found - margins fall back to raw cosine "
                "(run cohort_builder.py first for AS-norm margins)")
    per_rival = {}
    for w in rivals:
        if not clips.get(w):
            continue
        embs = l2_normalize(embed_batch(model, clips[w]), axis=1)
        per_rival[w] = (l2_normalize(embs.mean(axis=0)), embs)

    # Separability gate: a rival is armed only if its OWN clips lose the
    # margin contest decisively - a rival the embedding cannot separate from
    # the keyword would reject the keyword itself (sub-resolution, same
    # scope rule as effective homophones).
    kept, centroids, clip_embs = [], [], []
    for w, (centroid, embs) in per_rival.items():
        own_margin = rv.margins(embs, anchor, centroid[None, :], cohort)
        pos_margin = rv.margins(positives, anchor, centroid[None, :], cohort)
        if own_margin.max() > -rv.MIN_RIVAL_GAP:
            ui.warn(f"rival '{w}' dropped: sub-resolution "
                    f"(own-clip margin max {own_margin.max():+.3f})")
            continue
        if np.percentile(pos_margin, 10) < rv.MIN_POS_GAP:
            ui.warn(f"rival '{w}' dropped: keyword positives cannot beat it "
                    f"(pos margin p10 {np.percentile(pos_margin, 10):+.3f})")
            continue
        kept.append(w)
        centroids.append(centroid)
        clip_embs.append(embs)
    if not kept:
        ui.fail("every rival is sub-resolution - RAV cannot arm for this "
                "keyword; verification will pass detections through.")
        return
    centroids = np.stack(centroids)
    clip_embs = np.concatenate(clip_embs)

    ui.step("calibrating accept margin delta (synthetic-only, AS-norm) ...")
    delta, stats = rv.calibrate_delta(positives, clip_embs, anchor,
                                      centroids, cohort=cohort, log=ui.item)

    rv.save_rivals(npz_path, keyword, kept, centroids, delta, stats)
    ui.kv("rivals", ", ".join(kept))
    ui.kv("delta", f"{delta:+.3f}")
    ui.done(os.path.relpath(npz_path, PROJECT_ROOT), t0)


if __name__ == "__main__":
    main()
