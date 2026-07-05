# Siamese VMS - Zero-Shot Keyword Spotting in Live Video Streams

## What This Project Does

Give the system a **word** and a **live video/audio stream**, and it tells
you when that word gets spoken — without running full speech-to-text on the
live feed, and without any recording of that word in that exact voice.

You type a keyword like `"cloudy"`. The system:
1. **Imagines** how that word sounds using text-to-speech (multi-voice
   prototype anchor).
2. **Scans** live audio in sliding windows and scores each against the keyword
   embedding with Adaptive S-norm (AS-norm).
3. **Flags** windows whose score crosses a per-keyword calibrated threshold.

The result: zero-shot keyword spotting on live news audio — no enrollment
recording, no live ASR pass, no retraining per stream.

For a GUI that monitors a YouTube URL in real time, see
[`platform/README.md`](platform/README.md) (`python platform/gui.py`).

## Setup

```bash
cd Siamese_VMS_Project
pip install -r requirements.txt
```

Python 3.11 validated (see `requirements.txt` for exact package versions).
`moviepy` pulls in `imageio-ffmpeg`, which bundles its own ffmpeg binary —
no separate system ffmpeg install needed.

**Model checkpoints are already in this repo** (`checkpoints/*.pth`, a few
MB each) — the pipeline runs immediately with no training step. The
"Model Training" section further down documents how they were produced,
not something you need to redo.

**First run needs internet access**, even with every package installed:
`core/scoring.py` defaults `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` /
`HF_DATASETS_OFFLINE=1` for every run, so that a flaky connection can't
break a load of an already-cached model — but on a machine with nothing
cached yet, that same default makes the *first* run fail outright trying
to reach Hugging Face while offline. Do the first pipeline command with
those disabled, then leave them on afterward:
```bash
# Windows PowerShell
$env:HF_HUB_OFFLINE = "0"; $env:TRANSFORMERS_OFFLINE = "0"; $env:HF_DATASETS_OFFLINE = "0"
python pipeline/keyword_generator.py --keyword test
```
```bash
# bash
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0 python pipeline/keyword_generator.py --keyword test
```
This downloads SpeechT5 (TTS + HiFi-GAN), the CMU ARCTIC x-vectors dataset,
and WavLM into the local cache (~1-2 GB total); every run after that can
go back to the offline default.

**No audio is bundled in this repo** — `audios/Chunkset_D|E|F/transcripts.txt`
are kept as historical ground truth for the results below, but the
matching `.wav` chunks are not (large binary, and a live stream's content
isn't reproducible on demand anyway). Run `pipeline/downloader.py` to
pull your own chunks before anything else in the Usage Guide below.

## Why This Is Hard (The Core Problem)

Raw TTS compared directly to human speech mostly fails: a synthetic voice and
a real voice land in different regions of embedding space even when saying
the same word. We call this the **synthetic-to-real domain gap** — it was
the single biggest early failure mode (F1 = 0.00 on conversational speech
with a raw TTS anchor; see
[reports/EXPERIMENT_LOG.md](reports/EXPERIMENT_LOG.md)).

The fix: a **trained detector head** (attentive pooling on frozen WavLM
layer-10 features, trained with phonetic-confusable batches) closes the gap
in embedding space, so true keyword windows cross the calibrated threshold
directly — no voice conversion of the anchor and no third verification stage.
(Earlier versions of this pipeline closed the gap with kNN-VC voice
conversion instead; ablations showed the trained head makes it redundant —
see the note at the end of "Does It Actually Work?".)

## How It Works: The Two-Stage Pipeline

```mermaid
flowchart TD
    KW(["Keyword text<br/><b>e.g. 'cloudy'</b>"])
    STREAM(["Live stream audio"])
    RESULT(["Detections<br/><b>+ timestamps</b>"])

    subgraph S1["STAGE 1 — Anchor Building"]
        direction TB
        TTS["7 TTS voices<br/>SpeechT5 + HiFi-GAN"]
        CENTROID["L2-normalized centroid"]
        TTS --> CENTROID
    end

    subgraph S2["STAGE 2 — Detection"]
        direction TB
        SLIDE["Multi-scale windows<br/>0.6× / 0.8× / 1.0×, 50 ms hop"]
        EMBED["WavLM L10 frames<br/>+ attentive pooling head"]
        ASNORM["AS-norm vs impostor cohort<br/>→ threshold decision"]
        SLIDE --> EMBED --> ASNORM
    end

    KW --> TTS
    CENTROID --> EMBED
    STREAM --> SLIDE
    ASNORM --> RESULT

    classDef io fill:#f3e8ff,stroke:#9333ea,stroke-width:2px,color:#581c87,font-weight:bold
    classDef stage1 fill:#e0f2fe,stroke:#0284c7,stroke-width:2px,color:#0c4a6e
    classDef stage2 fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#78350f

    class KW,STREAM,RESULT io
    class TTS,CENTROID stage1
    class SLIDE,EMBED,ASNORM stage2

    style S1 fill:#f0f9ff,stroke:#0284c7,stroke-width:2px
    style S2 fill:#fffbeb,stroke:#d97706,stroke-width:2px
```

### Stage 1: Anchor Building

**Intuition:** synthesize the keyword with several voices so the anchor
captures the word, not any one speaker.

**Technical:** `pipeline/keyword_generator.py` builds a multi-voice TTS
prototype (SpeechT5 + HiFi-GAN: the 7 canonical CMU ARCTIC speakers by
default; random x-vectors and cross-speaker blends available via
`--n-random`/`--n-blend`), augments clips, embeds them, and averages a
centroid. A few voices are held out as calibration positives.

### Stage 2: Detection (WavLM + Attentive Head + AS-norm)

**Intuition:** slide windows across each chunk, score similarity to the
anchor, normalize against impostors, and accept windows above a threshold
fit on keyword-free stream audio.

**Production backend** (`SIAMESE_BACKEND=wavlm-trained`, default weights
`checkpoints/siamese_v3_best.pth`):

- **Backbone:** frozen `microsoft/wavlm-base-plus`, **layer 10** frame features
  (word identity peaks in mid layers, not the last layer).
- **Head:** trained attentive pooling + projection
  (`core/embedders.py`, `training/train_siamese_v3.py`) — sub-center ArcFace
  over MSWC + TTS bank with phonetic-confusable batches.
- **Scoring:** cosine similarity → **Adaptive S-norm** against a per-keyword
  cohort (`pipeline/cohort_builder.py`):

$$
s_{\text{norm}} = \frac{1}{2}\left(\frac{s - \mu_a}{\sigma_a} + \frac{s - \mu_w}{\sigma_w}\right)
$$

- **Detector:** `pipeline/detector.py` — multi-scale windows, 50 ms hop
  (must match calibration). Frame backends embed each chunk with one forward
  pass and pool windows from contextualized frames (~100× faster than
  per-window embedding).
- **Threshold:** `pipeline/calibrate.py` — max score on keyword-free windows
  embedded through the **same detector protocol** (chunk context, all scales,
  50 ms grid). Defaults to `--fa-percentile 100 --negatives 40000` (the
  trained-head operating point); negatives are drawn only from chunks that do
  not contain the keyword (transcript token check). The live platform
  (no pre-existing transcript) produces one itself — see
  [`platform/README.md`](platform/README.md#calibration-leakage-guard).

**Legacy baseline** (`SIAMESE_BACKEND=baseline`): frozen `wav2vec2-base` +
Phase-1 linear projection head (`core/siamese_model.py`,
`checkpoints/siamese_v1_best.pth`). Kept for comparison; not the production path.

## Does It Actually Work?

Set D validation (Sky News weather bulletin, 10 unique chunks,
`audios/Chunkset_D/transcripts.txt` ground truth) with `wavlm-trained` and
aligned p100 calibration:

| Keyword | Difficulty | Precision | Recall | F1 |
|---|---|---|---|---|
| russia | rare (1 chunk) | 1.00 | 1.00 | 1.00 |
| weather | rare (1 chunk) | 1.00 | 1.00 | 1.00 |
| scotland | frequent (3 chunks) | 1.00 | 1.00 | 1.00 |
| ireland | frequent (3), near-homophone ("Island") | 1.00 | 1.00 | 1.00 |
| brighten | rare (1 chunk) | 1.00 | 1.00 | 1.00 |
| **Micro-avg (50 decisions)** | | **1.00** | **1.00** | **1.00** |

Additional ad-hoc runs (same chunk set): `cloudy` (2 true chunks) F1 = 1.00;
`outbreaks` F1 = 0.67 (one chunk-boundary FN documented in the experiment log).

Full history, ablations, and failure analyses:
[reports/EXPERIMENT_LOG.md](reports/EXPERIMENT_LOG.md),
[reports/SIAMESE_PROGRESS_REPORT.md](reports/SIAMESE_PROGRESS_REPORT.md).

Pipeline parameter defaults (voice count, holdout, stream-window count, TTS
distractors) come from a one-at-a-time ablation over the same Set D keywords:
`ablation_study/` — the `combined_best` config cuts wall time ~70% vs. the
original search-time defaults with no F1 loss (`ablation_study/results/`).

**Retired: kNN-VC anchor conversion.** Earlier pipeline versions re-voiced
the TTS anchor into the stream's own voice with
[kNN-VC](https://github.com/bshall/knn-vc) to close the domain gap. With the
trained v3 head, ablations found no F1 benefit on 9/9 keywords across two
chunk sets — including `administration` vs. its near-homophone
`immigration`, the confusable case voice conversion existed to protect
(`ablation_study/results/knnvc_ablation.jsonl`). The kNN-VC stage was
removed; the full implementation is preserved on the
`archive/knnvc-pipeline` branch.

## Configuration

Set these before running the pipeline (PowerShell example):

```powershell
$env:SIAMESE_BACKEND = "wavlm-trained"
$env:SIAMESE_V3_WEIGHTS = "checkpoints/siamese_v3_best.pth"
```

| Variable | Default | Purpose |
|---|---|---|
| `SIAMESE_BACKEND` | `baseline` | `wavlm-trained` (production), `wavlm`, or `baseline` |
| `SIAMESE_WEIGHTS` | `checkpoints/siamese_v1_best.pth` | Checkpoint for the `baseline` backend only - **not** read by `wavlm-trained` |
| `SIAMESE_V3_WEIGHTS` | `checkpoints/siamese_v3_best.pth` | Attentive-head weights for `wavlm-trained` |
| `SIAMESE_PROJECT_ROOT` | project root | Redirect all data paths (used by `platform/`) |
| `SIAMESE_AUDIO_DIR` | `audios/` | Chunk set to score against (e.g. `audios/Chunkset_E`) |
| `SIAMESE_BACKBONE` | `microsoft/wavlm-base-plus` | SSL model for WavLM backends |
| `SIAMESE_LAYER` | `10` | Transformer layer for frame features |

Non-baseline backends write suffixed artifacts (e.g.
`keywords/cloudy_anchor_wavlm10ft.npz`) so runs do not overwrite each other.

## Project Layout

```
Siamese_VMS_Project/
├── core/           siamese_model.py (Phase-1 head), embedders.py (WavLM backends),
│                   scoring.py (AS-norm, paths, backends), augment_utils.py
├── pipeline/       offline research pipeline (run order below)
├── platform/       live YouTube monitoring GUI (isolated under platform/data/)
├── training/       AWS training scripts: v1 triplet, v2 GRL, v3 attentive head
├── checkpoints/    siamese_v1_best.pth, siamese_v2_best.pth, siamese_v3_best.pth
├── keywords/       generated anchors, cohorts, calibrations (gitignored, rebuildable)
├── audios/         Chunkset_D/, Chunkset_E/, Chunkset_F/ - chunks +
│                   transcripts.txt each (select one via SIAMESE_AUDIO_DIR);
│                   videos/ mirrors the chunk sets
├── logs/           detection JSON + archive of historical runs
└── reports/        experiment log + progress report
```

### Pipeline scripts (run from project root)

| Step | Script | Output |
|---|---|---|
| 1 | `downloader.py` | `audios/`, `videos/` |
| 2 | `transcribe_chunks.py` | `audios/.../transcripts.txt` |
| 3 | `keyword_generator.py` | TTS anchor + variants |
| 4 | `cohort_builder.py` | `cohort_<kw>*.npz` |
| 5 | `calibrate.py` | `*_calibration*.json` |
| 6 | `detector.py` | `logs/detections_<kw>.json` |
| 7 | `validate_detection.py` | P / R / F1 vs transcripts |

## Model Training (Offline, AWS)

Three training generations live under `training/`:

| Phase | Script | What it trains | Checkpoint |
|---|---|---|---|
| 1 | `train_siamese_v1.py` | wav2vec2-base (frozen) + linear projection head, triplet loss on MSWC | `siamese_v1_best.pth` |
| 2 | `train_siamese_v2.py` | Same backbone + GRL domain adversary on the head (did not fix ranking inversions) | `siamese_v2_best.pth` |
| 3 | `train_siamese_v3.py` | **Attentive pooling head** on frozen WavLM L10, sub-center ArcFace, phonetic-confusable batches | `siamese_v3_best.pth` |

Launch helpers: `deploy_phase1.ps1`, `deploy_phase2.ps1`, `deploy_phase3.ps1`.

## Usage Guide

Run from the project root. Pass `--keyword` explicitly (or create
`selected_keyword.txt` in the project root).

```powershell
cd Siamese_VMS_Project
$env:SIAMESE_BACKEND = "wavlm-trained"
$env:SIAMESE_V3_WEIGHTS = "checkpoints/siamese_v3_best.pth"
```

1. **Download stream chunks** (content-hash dedup):
   ```bash
   python pipeline/downloader.py
   ```
2. **Transcribe chunks** (ground truth for validation + the calibration
   leakage guard):
   ```bash
   python pipeline/transcribe_chunks.py
   ```
3. **TTS prototype anchor** (7 canonical voices, holdout 4 — defaults tuned by
   `ablation_study/`, no accuracy loss vs. the original 17-voice search):
   ```bash
   python pipeline/keyword_generator.py --keyword cloudy
   ```
4. **Impostor cohort** (50 stream windows, TTS distractors off by default):
   ```bash
   python pipeline/cohort_builder.py --keyword cloudy
   ```
5. **Calibrate threshold** (aligned detector protocol, keyword-free negatives):
   ```bash
   python pipeline/calibrate.py --keyword cloudy
   ```
6. **Detect:**
   ```bash
   python pipeline/detector.py --keyword cloudy
   ```
7. **Validate:**
   ```bash
   python pipeline/validate_detection.py --keyword cloudy
   ```

## Related Approaches in This Repo

| Folder | Method |
|---|---|
| `Correlation_VMS_Project/` | Original TTS + cross-correlation baseline |
| `PhonMatchNet_VMS_Project/` | G2P open-vocabulary phoneme matching (no audio anchor) |
| **Siamese_VMS_Project/** | This project — Siamese embedding + trained WavLM detector |
