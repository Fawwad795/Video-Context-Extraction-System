"""Convert the TTS anchor clips into the live stream's own voice with kNN-VC.

Step-1 experiment for closing the synthetic-to-real domain gap: the frozen
wav2vec2 backbone embeds TTS and real speech in different regions of the
space, so a TTS centroid anchor is compared across domains and confusable
real words can outrank the true keyword ("appropriate" > "penalty").

kNN-VC (Baas et al., Interspeech 2023 - github.com/bshall/knn-vc) converts
any source utterance into a target voice with zero training: WavLM features
of the source are replaced frame-by-frame with the mean of their k nearest
neighbours in the reference audio, then vocoded with a prematched HiFi-GAN.
Because every output frame is assembled from *real stream frames*, the
converted anchor lies in the deployment domain by construction - same
speakers, room, codec - while keeping the keyword's phonetic content.

Leakage guard: chunks whose Whisper transcript contains the keyword are
excluded from the kNN reference set, so the anchor cannot copy frames of an
actual utterance of the keyword.

Reads:   keywords/<kw>_variants/*.wav   (cached TTS syntheses)
         audios/live_*.wav + audios/transcripts.txt (reference pool)
Writes:  keywords/<kw>_variants_knnvc/*.wav
         keywords/<kw>_anchor.npz  (same format as keyword_generator.py;
         the previous anchor is backed up to <kw>_anchor_tts.npz)

After this, re-run: cohort_builder.py, calibrate.py, detector.py.
"""

import argparse
import glob
import os
import re
import shutil

import librosa
import numpy as np
import soundfile as sf
import torch

# Shared modules (scoring, siamese_model, augment_utils) live in ../core
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "core"))

from scoring import (PROJECT_ROOT, SAMPLE_RATE, embed_batch, l2_normalize,
                     list_chunk_audios, load_siamese_model)

MIN_CLIP_SECONDS = 0.15


def load_wav_tensor(path):
    """Load a wav as a (1, T) float32 16 kHz tensor.

    torchaudio>=2.9 routes .load() through torchcodec (an optional FFmpeg
    binding not installed here), so we decode with librosa and hand kNN-VC a
    tensor - get_features/get_matching_set accept tensors directly.
    """
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return torch.from_numpy(y.astype(np.float32)).unsqueeze(0)


def keyword_free_chunks(keyword):
    """Live chunks whose transcript does not contain the keyword."""
    transcript_path = os.path.join(PROJECT_ROOT, "audios", "transcripts.txt")
    if not os.path.exists(transcript_path):
        transcript_path = os.path.join(PROJECT_ROOT, "transcripts.txt")
    contains_kw = {}
    if os.path.exists(transcript_path):
        current = None
        for line in open(transcript_path, encoding="utf-8"):
            m = re.match(r"\[(live_\d+\.wav)\]", line.strip())
            if m:
                current = m.group(1)
                contains_kw.setdefault(current, False)
            elif current:
                tokens = set(re.findall(r"[a-z']+", line.lower()))
                if keyword.lower() in tokens:
                    contains_kw[current] = True
    files = []
    for f in list_chunk_audios():
        name = os.path.basename(f)
        if contains_kw.get(name, False):
            print(f"  excluding {name} from reference (contains '{keyword}')")
        else:
            files.append(f)
    return files


def main():
    ap = argparse.ArgumentParser(description="kNN-VC domain-converted anchor.")
    ap.add_argument("--keyword", default=None, help="defaults to selected_keyword.txt")
    ap.add_argument("--topk", type=int, default=4, help="kNN neighbours per frame")
    ap.add_argument("--holdout", type=int, default=6,
                    help="converted voices held out as calibration positives")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    keyword = args.keyword
    if keyword is None:
        kw_file = os.path.join(PROJECT_ROOT, "selected_keyword.txt")
        if not os.path.exists(kw_file):
            print("No --keyword given and selected_keyword.txt not found.")
            return
        keyword = open(kw_file).read().strip()
    print(f"Building kNN-VC converted anchor for keyword: '{keyword}'")

    variants_dir = os.path.join(PROJECT_ROOT, "keywords", f"{keyword}_variants")
    src_wavs = sorted(glob.glob(os.path.join(variants_dir, "*.wav")))
    if not src_wavs:
        print(f"No TTS variants in {variants_dir} - run keyword_generator.py first.")
        return

    ref_wavs = keyword_free_chunks(keyword)
    if not ref_wavs:
        print("No keyword-free reference chunks available. Aborting.")
        return
    ref_seconds = sum(librosa.get_duration(path=f) for f in ref_wavs)
    print(f"Reference pool: {len(ref_wavs)} chunks, {ref_seconds:.0f}s of stream audio")

    print("Loading kNN-VC (WavLM-Large + prematched HiFi-GAN) via torch.hub...")
    knn_vc = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True,
                            trust_repo=True, pretrained=True, device="cpu")
    ref_tensors = [load_wav_tensor(f) for f in ref_wavs]
    matching_set = knn_vc.get_matching_set(ref_tensors)
    print(f"Matching set: {matching_set.shape[0]} WavLM frames")

    out_dir = os.path.join(PROJECT_ROOT, "keywords", f"{keyword}_variants_knnvc")
    os.makedirs(out_dir, exist_ok=True)

    clips = []  # (name, trimmed converted audio)
    for i, src in enumerate(src_wavs):
        name = os.path.basename(src)
        out_path = os.path.join(out_dir, name)
        if os.path.exists(out_path):
            audio, _ = librosa.load(out_path, sr=SAMPLE_RATE)
        else:
            with torch.no_grad():
                query_seq = knn_vc.get_features(load_wav_tensor(src))
                wav = knn_vc.match(query_seq, matching_set, topk=args.topk)
            audio = wav.squeeze().cpu().numpy().astype(np.float32)
            sf.write(out_path, audio, samplerate=SAMPLE_RATE)
        trimmed, _ = librosa.effects.trim(audio, top_db=30)
        if len(trimmed) < MIN_CLIP_SECONDS * SAMPLE_RATE:
            print(f"  [{i + 1}/{len(src_wavs)}] {name}: too short after trim - skipped")
            continue
        clips.append((name, trimmed.astype(np.float32)))
        print(f"  [{i + 1}/{len(src_wavs)}] {name}: {len(trimmed) / SAMPLE_RATE:.2f}s converted")

    # Same degenerate-clip guard as keyword_generator.py
    median_len = np.median([len(c) for _, c in clips])
    clips = [(n, c) for n, c in clips if len(c) <= 1.8 * median_len]
    if len(clips) <= args.holdout + 2:
        print("Not enough usable converted clips. Aborting.")
        return

    window_samples = int(np.median([len(c) for _, c in clips]))

    # No augmentation: the clips are already in the deployment domain; adding
    # synthetic reverb/noise back would reintroduce mismatch.
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(clips))
    holdout_idx = set(order[:args.holdout].tolist())
    centroid_audio = [c for i, (_, c) in enumerate(clips) if i not in holdout_idx]
    positive_audio = [c for i, (_, c) in enumerate(clips) if i in holdout_idx]

    model = load_siamese_model()
    print(f"Embedding {len(centroid_audio)} centroid clips + "
          f"{len(positive_audio)} held-out positives...")
    centroid_embs = embed_batch(model, centroid_audio)
    positives = embed_batch(model, positive_audio)

    centroid = l2_normalize(centroid_embs.mean(axis=0))
    cos_to_centroid = centroid_embs @ centroid
    keep = cos_to_centroid >= 0.5
    if 0 < (~keep).sum() < 0.3 * len(centroid_embs):
        print(f"Trimming {(~keep).sum()} outlier embeddings (cos < 0.5).")
        centroid = l2_normalize(centroid_embs[keep].mean(axis=0))
        cos_to_centroid = centroid_embs[keep] @ centroid
    print(f"Centroid cohesion: cos mean={cos_to_centroid.mean():.3f} "
          f"min={cos_to_centroid.min():.3f}")

    anchor_path = os.path.join(PROJECT_ROOT, "keywords", f"{keyword}_anchor.npz")
    backup_path = os.path.join(PROJECT_ROOT, "keywords", f"{keyword}_anchor_tts.npz")
    if os.path.exists(anchor_path) and not os.path.exists(backup_path):
        shutil.copy2(anchor_path, backup_path)
        print(f"Baseline TTS anchor backed up: {backup_path}")

    np.savez(anchor_path,
             centroid=centroid.astype(np.float32),
             positives=positives.astype(np.float32),
             window_samples=np.int64(window_samples),
             keyword=np.str_(keyword))
    print(f"kNN-VC anchor saved: {anchor_path}")
    print(f"  clips used: {len(clips)} ({len(centroid_audio)} centroid, "
          f"{len(positive_audio)} held out)  |  window: "
          f"{window_samples / SAMPLE_RATE:.2f}s")
    print("Next: python cohort_builder.py && python calibrate.py && python detector.py")


if __name__ == "__main__":
    main()
