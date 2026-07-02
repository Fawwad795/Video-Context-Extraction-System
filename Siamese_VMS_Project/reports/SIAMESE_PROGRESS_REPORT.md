# Siamese KWS — Progress Report
**Project:** Video Context Extraction System  
**Date:** 2026-07-02  
**Scope:** Siamese triplet audio-anchor approach only (Phases 1 & 2). For the G2P/PhonMatchNet approach see `PhonMatchNet_VMS_Project/`.

---

## Goal

Zero-shot audio keyword spotting in live YouTube news streams using a Siamese triplet network. The user provides a keyword → the system synthesizes an audio anchor from TTS, embeds it, and detects matching windows in live stream chunks via distance-based scoring.

---

## Architecture

```
keyword text ──► TTS (30 voices) ──► centroid anchor embedding
                                              │
live audio windows ──► wav2vec2-base (frozen) ──► 256-d projection head ──► L2 distance ──► AS-norm score
```

- **Backbone:** `facebook/wav2vec2-base` (frozen throughout all phases)
- **Projection head:** Linear → ReLU → Linear → 256-d L2-normalized embedding
- **Loss:** `TripletMarginLoss(margin=1.0)` on (anchor-TTS, positive-TTS, negative-TTS) triplets
- **Training data:** MLCommons MSWC (ml_spoken_words), 5.1M clips, AWS EBS 1000GB volume
- **Inference:** cosine similarity → AS-norm score → threshold → detection

---

## Phase 1 — Multi-Voice Centroid Anchor + AS-norm

### What was built (`Siamese_VMS_Project/`)

**`keyword_generator.py`** — anchor synthesis:
- 30 TTS voices: 7 CMU ARCTIC speakers + 13 random x-vectors + 10 cross-speaker blends
- Augmentation (`augment_utils.py`): pitch shift, time stretch, RIR reverb, band-limit, additive noise
- Degenerate clip rejection: drop clips > 1.8× median duration
- Two-pass centroid: embed all clips → drop embeddings with cosine < 0.5 from mean → re-average
- 6 voices held out as calibration positives

**`scoring.py`** — shared scoring utilities:
- `load_siamese_model()` — loads backbone + projection head; honors `SIAMESE_WEIGHTS` env var
- `embed_batch()` — L2-normalized embeddings via frozen wav2vec2
- `topk_stats(scores, k)` → (mean, std) of top-k cohort scores
- `asnorm_windows(score, anchor_emb, window_emb, cohort)` — adaptive S-norm:  
  `s_norm = 0.5 * ((s - μ_a)/σ_a + (s - μ_w)/σ_w)` using top-k closest impostors per trial
- `load_cohort(keyword)` → per-keyword `cohort_<keyword>.npz`
- `sample_stream_windows()` — cohort sampling from stream audio
- `AUDIO_DIR` honors `SIAMESE_AUDIO_DIR` env var

**`cohort_builder.py`** — builds per-keyword cohort `.npz` from stream audio (windows of stream speech without the keyword)

**`calibrate.py`** — percentile-FA threshold:
- Score keyword anchor vs negative cohort
- `threshold = percentile(negative_scores, fa_percentile)` (e.g. 99th)
- Stores per-keyword threshold in `keywords/<kw>_calibration.json`
- **Note:** σ-rule (mean + k×std) fails — negative AS-norm scores are left-skewed (mean −6.3, std 5.5, max only 2.1); switched to empirical percentile

**`detector.py`** — detection:
- Multi-scale windows: 0.6×, 0.8×, 1.0× of anchor duration (`--scales` argparse flag)
- AS-norm scoring per window
- Loads calibrated threshold from `keywords/<kw>_calibration.json`
- JSON output to `logs/detections_<kw>.json`

**`validate_detection.py`** — F1 evaluation against ground-truth timestamps

**`transcribe_chunks.py`** — Whisper ASR to identify which chunks contain the keyword

### Phase 1 Results

| Keyword | Domain | F1 | Notes |
|---------|--------|----|-------|
| absolutely | Scripted news speech | 0.57 | Old system: 0 matches |
| heat | Scripted news speech | 0.80 | Old system: 0 matches |
| penalty | Conversational debate speech | **0.00** | Confusable "appropriate" outranked true keyword (raw cosine 0.984 vs 0.949) |

**Conclusion:** Phase 1 scoring machinery works end-to-end on scripted speech. Ranking fails on conversational speech — confusable words outscore the true keyword. This is an embedding-quality failure; no threshold can fix it.

**Root cause diagnosed:** Linear probe separates TTS from real hidden states with 98% accuracy (arXiv:2408.10463). The wav2vec2 frozen backbone embeds TTS and real speech in different regions — the anchor is always in the "TTS cluster" and real speech windows fall in the "real cluster", so distances are unreliable.

---

## Phase 2 — GRL Domain-Adversarial Retrain

### What was built

**`train_siamese_v2.py`** — domain-adversarial training:
- `GradReverse` class — gradient reversal layer (GRL)
- `domain_head` — 2-layer MLP domain classifier attached after projection head
- Combined loss: `L = (1 - β) × TripletLoss + β × BCE(domain | GRL(embedding))`
- β ramp over `beta_ramp` epochs (avoids early training instability)
- `eval_cross_domain_auc()` — TTS centroid vs real human clips on held-out eval words
- Warmstart from `best_siamese_model.pth`

**Infrastructure (AWS g6.xlarge, NVIDIA L4):**
- `tts_bank.py` — generated 1650 words × 5 voices = 8085 TTS clips; 150 eval words held out
- `dataset_v2.py` — cross-domain triplets + domain labels for GRL
- AWS instance: i-0107876d52d64b5f6, key MyKey.pem, ec2-user
- MSWC dataset cached at 288GB on EBS 1000GB volume

### Gotchas Hit During Training

| Issue | Fix |
|-------|-----|
| `sentencepiece` missing on AWS | `pip3 install sentencepiece` |
| `transformers>=4.56` blocks `torch.load` with `torch<2.6` (SpeechT5) | `use_safetensors=True` |
| GRL with β=0.3 hurt performance | Frozen backbone can't achieve domain invariance; domain-acc stuck at 0.99 |

### Phase 2 Results

| Run | Config | Best AUC | Notes |
|-----|--------|----------|-------|
| v1 baseline | No GRL | 0.8213 | Starting point |
| Run 1 | β=0.3 | 0.76–0.80 | GRL HURT; killed at epoch 19 |
| Run 2 | β=0.05 | 0.8311 (ep 28) | Marginal gain over baseline |
| Run 1 early | mixing-only checkpoint | 0.8381 | Best single checkpoint |

**"penalty" chunk test:**
- v1 (baseline): F1=0
- v2-run1: F1=0.36 (recall=1.0, precision=0.22) — recalls the keyword but floods with FPs
- v2-run2: F1=0 — confusables still outrank true keyword

**Saved checkpoints (local):**
- `siamese_v2_best.pth` (run2, β=0.05)
- `siamese_v2_run1_beta30_best.pth` (run1)
- `scoring.py` honors `SIAMESE_WEIGHTS` env var to select checkpoint

**Conclusion:** Head-only retraining with GRL hit an architecture ceiling (~+1 pt AUC). The frozen wav2vec2 backbone cannot be made domain-invariant via the head alone — the domain information is too deeply encoded. Next lever to try: unfreeze top wav2vec2 transformer layers during GRL training. Alternatively, abandon audio anchors entirely (→ Phase 3 / PhonMatchNet).

---

## Architecture Ceiling Analysis

The fundamental problem is structural:

1. **TTS vs real embedding gap** — 98% linear probe accuracy separating TTS from real speech means the backbone encodes domain identity, not just phonetic content. The anchor always lives in the TTS subspace; real speech windows live in the real subspace; L2 distances across subspaces are unreliable.

2. **Ranking inversions** — "appropriate" scores higher than "penalty" in raw cosine (0.984 vs 0.949). This isn't a threshold problem — the impostor literally outranks the true keyword. AS-norm partially compensates but can't flip rankings when the backbone itself is confused.

3. **Head-only GRL ceiling** — GRL can only remove domain information from the 256-d head output, but the frozen backbone already encodes it in ways the small head can't undo.

**What would actually help:**
- Unfreeze top 2–4 wav2vec2 transformer layers during GRL training (gives the GRL capacity to reshape representations)
- Or abandon audio-anchor enrollment entirely (Phase 3: query-by-text with PhonMatchNet)

---

## File Map

```
Siamese_VMS_Project/
├── scoring.py              # AS-norm, embedding, cohort utilities
├── keyword_generator.py    # 30-voice TTS centroid anchor
├── cohort_builder.py       # per-keyword cohort_<kw>.npz
├── calibrate.py            # percentile-FA threshold → keywords/<kw>_calibration.json
├── detector.py             # multi-scale windows + AS-norm + calibrated threshold
├── validate_detection.py   # F1 evaluation
├── transcribe_chunks.py    # Whisper ASR (ground truth)
├── train_siamese_v2.py     # GRL domain-adversarial training (Phase 2)
├── augment_utils.py        # pitch/stretch/RIR/noise augmentation
├── denoise_chunks.py       # noisereduce front-end (experimental)
├── downloader.py           # live stream chunk downloader
└── logs/                   # detection JSON outputs (gitignored)
```

**Gitignored:**
- `cohort_*.npz` — per-keyword cohort embeddings (rebuildable)
- `keywords/` — generated TTS anchors and calibration files
- `logs/detections_*.json` — detection outputs
- `audios_prev/`, `videos_prev/`, `audios_denoised/`, `pretrained_models/`

---

## Key Lessons

- **σ-rule threshold fails on left-skewed negatives** — use empirical percentile (99th FA) instead
- **Per-keyword cohort files required** — a single `cohort.npz` collides across keywords with different window lengths; use `cohort_<keyword>.npz`
- **Denoising helps cosine rank but not ranking inversions** — `noisereduce` (prop_decrease ~0.6–0.7) improves elon rank #9→#5, penalty #5→#2, but impostors still rank above true keyword; keep as cheap complement, not architectural fix
- **Multi-scale windows needed** — spoken keyword duration varies (0.40s) vs TTS anchor (0.70s); 0.6×/0.8×/1.0× scales improve recall
- **GRL needs unfrozen backbone to work** — frozen backbone = domain-acc stuck at 0.99, GRL can't achieve invariance

---

## Current State

The Siamese approach is **complete but capped**. Phase 1 and Phase 2 are both committed to `main` under `Siamese_VMS_Project/`. The approach was superseded by PhonMatchNet (Phase 3) which beats it on both LibriPhrase benchmark (28% vs 44% LP-Hard EER) and on the "penalty"/"elon" VMS test chunks (F1=1.00 vs F1=0/0.20).

The Siamese code is preserved as the comparison baseline. The recommended next step for the Siamese line (if revisiting) would be unfreezing top wav2vec2 layers in `train_siamese_v2.py`.
