# Siamese VMS — Zero-Shot Keyword Spotting in Live Video Streams

## What this project does (plain English)

Give the system a **word** and a **live video/audio stream**, and it tells you
*when that word gets spoken* — without ever running a full speech-to-text
transcript, and without any recording of that word said in that exact voice.

You type a keyword like `"washington"`. The system:
1. **Imagines** how that word sounds using text-to-speech (many synthetic
   voices, not a real recording).
2. **Repaints** that imagined sound into something that sounds like it came
   from *this specific stream* (same speaker/mic/room tone).
3. **Scans** the live audio in small time-slices, comparing each slice's
   "sound fingerprint" to the keyword's fingerprint.
4. **Double-checks** every promising slice by asking "does this actually
   contain the keyword's spoken sounds?" — not just "does it sound similar?"

The result: it can catch a keyword nobody ever recorded, spoken by a
newsreader nobody trained the system on, inside a live stream it has never
seen — with no live transcription step slowing things down.

## Why this is hard (the core problem)

The obvious approach — synthesize the keyword with TTS, then check "does
this window of live audio sound like my synthetic clip?" — mostly doesn't
work. A computer-generated voice and a real human voice look completely
different to a neural network, even when they're saying the exact same
word. It's like trying to recognize a friend's face using only a
cartoon drawing of them: the *shape* is the same, but a face-recognition
model trained on real photos may not see the resemblance at all. We call
this the **synthetic-to-real domain gap**, and it was the single biggest
failure mode in this project (see [Reports/EXPERIMENT_LOG.md](Reports/EXPERIMENT_LOG.md)
for the measured failures: F1 = 0.00 on conversational speech with a raw
TTS anchor).

## How it works (the three-stage pipeline)

```
  keyword text                    live stream audio (10 chunks)
       │                                    │
       ▼                                    │
 ┌─────────────┐                            │
 │  1. ANCHOR  │  many TTS voices  ──►  kNN  │
 │  BUILDING   │  say the keyword    voice-  │
 │             │                     convert │
 └──────┬──────┘  into the stream's own voice│
        │                                    │
        ▼                                    ▼
 ┌─────────────────────────────────────────────────┐
 │  2. DETECTION - Siamese network + AS-norm        │
 │  slide a small window across the audio, embed    │
 │  each slice, compare to the anchor, normalize     │
 │  the score against a cohort of "not the keyword"  │
 └──────────────────────┬────────────────────────────┘
                         │  candidate time-stamps
                         ▼
 ┌─────────────────────────────────────────────────┐
 │  3. VERIFICATION - phoneme check                  │
 │  decode each candidate's actual speech sounds      │
 │  (CTC phoneme recognizer) and require them to      │
 │  match the keyword's phoneme sequence               │
 └──────────────────────┬────────────────────────────┘
                         ▼
                confirmed detections + timestamps
```

### Stage 1 — Anchor building (bridging the domain gap)

**Plain English:** we can't record a human saying every possible keyword in
advance, so we fake it with text-to-speech — but a *robot voice* comparison
against *real human* audio doesn't work well (the domain gap above). So we
take the extra step of re-voicing the synthetic clip so it sounds like it
came from the actual stream.

**Technical:** `pipeline/keyword_generator.py` synthesizes the keyword with
~30 voices (SpeechT5 + HiFi-GAN: the 7 canonical CMU ARCTIC speakers, random
utterance-level x-vectors, and cross-speaker x-vector blends), augments each
clip (pitch/tempo/noise/reverb), embeds everything with the Siamese model,
and averages into an L2-normalized **centroid anchor**. Then
`pipeline/convert_anchor_knnvc.py` runs each TTS clip through
**[kNN-VC](https://github.com/bshall/knn-vc)** (WavLM features + k-nearest-neighbor
regression + a prematched HiFi-GAN vocoder, no training required), using
non-keyword segments of the live stream as the voice reference. Every output
frame is literally reassembled from real stream audio, so the converted
anchor sits in the *real* acoustic domain instead of the *synthetic* one.
Chunks that contain the keyword are excluded from the reference pool to
avoid leaking the answer into the anchor.

### Stage 2 — Detection (Siamese network + adaptive scoring)

**Plain English:** slide a small window across the audio, ask a neural
network "does this sound like the keyword?", and flag the windows that do —
but calibrate "does" relative to what background speech normally scores, not
a fixed number, so accents/pace/room acoustics don't need re-tuning by hand.

**Technical:** the backbone is `facebook/wav2vec2-base` (frozen) feeding a
trained 256-d L2-normalized projection head (`core/siamese_model.py`,
`core/scoring.py`). `pipeline/detector.py` slides windows at multiple
scales (0.6x/0.8x/1.0x of the anchor duration, since spoken pace varies) and
scores each by cosine similarity, then applies **Adaptive S-norm**:

```
s_norm = 0.5 * ( (s - mu_a)/sd_a  +  (s - mu_w)/sd_w )
```

where `(mu_a, sd_a)` and `(mu_w, sd_w)` are the top-k closest-impostor
statistics from a per-keyword cohort (`pipeline/cohort_builder.py`) on the
anchor side and the window side respectively. The accept threshold is fitted
per keyword by `pipeline/calibrate.py` as an empirical false-alarm
percentile on that cohort (not a hand-picked distance). The detector also
keeps its top-8 non-maximum-suppressed windows per chunk as "candidates"
regardless of threshold, so stage 3 can rescue a true detection that scored
just under the line.

### Stage 3 — Verification (phoneme precision filter)

**Plain English:** the detector's "sounds similar" test can be fooled by
unrelated phrases that happen to have a similar overall shape (e.g. two
different sentences that are both fast and low-pitched). So every candidate
gets a second, completely different check: what *speech sounds* does it
actually contain, and do they match the keyword's?

**Technical:** `pipeline/verify_detections.py` CTC-decodes each candidate
window with `facebook/wav2vec2-lv-60-espeak-cv-ft` into an IPA phoneme
sequence, and compares it (infix edit distance, free ends) against
references decoded from the anchor clips. This model is trained to predict
*phonemes*, not speaker or recording identity, so its output is invariant to
the exact domain gap that Stage 1 works around — TTS and kNN-VC-converted
clips of the same word decode to identical phoneme strings. The accept
threshold is the midpoint between the negative (random stream window)
phone-similarity percentile and the references' own leave-one-out
self-similarity, floored at 0.5.

## Does it actually work? (validation results)

The clearest evidence is the most recent validation run, on a chunk set
(Sky News weather bulletin) the pipeline had never been tuned against, with
four keywords deliberately chosen to span difficulty — not cherry-picked:

| Keyword | Difficulty | Precision | Recall | F1 |
|---|---|---|---|---|
| russia | rare (1 true chunk), distinct | 1.00 | 1.00 | 1.00 |
| weather | rare (1 true chunk), distinct | 1.00 | 1.00 | 1.00 |
| scotland | frequent (3 true chunks) | 0.75 | 1.00 | 0.86 |
| ireland | frequent (3), near-homophone ("Island") stress test | 1.00 | 0.67 | 0.80 |
| **Aggregate (40 chunk decisions)** | | **0.875** | **0.875** | **0.875** |

Both remaining errors trace to a single borderline chunk, not a systemic
failure — full breakdown, per-chunk detail, and every historical experiment
(including the F1 = 0.00 domain-gap failures that motivated stages 1 and 3)
are in [Reports/EXPERIMENT_LOG.md](Reports/EXPERIMENT_LOG.md).

## Project Layout

```
Siamese_VMS_Project/
├── core/         shared modules: siamese_model.py (network), scoring.py (embedding,
│                 AS-norm, cohort utils), augment_utils.py (audio augmentation)
├── pipeline/     the detection pipeline, in run order: downloader.py → transcribe_chunks.py
│                 (or transcriber.py) → keyword_generator.py → convert_anchor_knnvc.py
│                 (recommended) → cohort_builder.py → calibrate.py → detector.py →
│                 verify_detections.py (phoneme precision filter) → validate_detection.py
│                 (+ denoise_chunks.py utility)
├── training/     offline model training (AWS): train_siamese.py (Phase 1),
│                 train_siamese_v2.py + dataset_v2.py + tts_bank.py (Phase 2 GRL),
│                 deploy_*.ps1 launch scripts
├── checkpoints/  trained weights (best_siamese_model.pth = Phase 1 baseline;
│                 siamese_v2_*.pth = Phase 2 runs; select via SIAMESE_WEIGHTS env var)
├── keywords/     generated anchors, calibrations, cohorts (per keyword)
├── audios/ videos/  downloaded live-stream chunks + transcripts.txt (ground truth)
├── logs/         detection outputs (logs/archive/ = retired chunk-set runs)
└── Reports/      EXPERIMENT_LOG.md (results ledger) + SIAMESE_PROGRESS_REPORT.md (narrative)
```

Run pipeline scripts from the project root, e.g. `python pipeline/detector.py --keyword washington`.

## Model architecture (Siamese network)

- **Feature Extractor (Backbone):** `facebook/wav2vec2-base`, frozen — a
  pre-trained self-supervised speech model that turns raw audio into robust
  acoustic features.
- **Metric Layer (Projection Head):** a trained linear projection head that
  compresses the 768-dimensional wav2vec2 output into a 256-dimensional
  L2-normalized embedding space.
- **Loss Function:** `TripletMarginLoss(margin=1.0)` — trained on an anchor
  word, a positive sample (the same word, different speaker), and a
  negative sample (a different word), pulling same-word embeddings together
  and pushing different-word embeddings apart.
- **Training data:** `MLCommons/ml_spoken_words` — 5.1 million short human
  speech clips, on AWS `g6.xlarge` (NVIDIA L4) with a 500 GB EBS volume for
  dataset caching. The Phase-1 baseline converged to a validation loss of
  ~0.29 over 50 epochs. A Phase-2 run added a gradient-reversal
  domain-adversarial loss to reduce TTS-vs-real sensitivity in the backbone
  itself (`training/train_siamese_v2.py`); see `Reports/SIAMESE_PROGRESS_REPORT.md`
  for that experiment's outcome.

## Usage Guide

Run from the project root. Most scripts default `--keyword` to
`selected_keyword.txt` if you omit it.

1. **Download & chunk a live stream** (dedupes on content hash, not just the
   segment URL, since YouTube re-signs segment URLs on every poll):
   ```bash
   python pipeline/downloader.py
   ```
2. **Get ground-truth transcripts** for every chunk (used later for
   validation and, optionally, to auto-pick a keyword):
   ```bash
   python pipeline/transcribe_chunks.py
   ```
3. **Build the multi-voice TTS prototype anchor:**
   ```bash
   python pipeline/keyword_generator.py --keyword washington
   ```
4. **Convert the anchor into the stream's own voice** (recommended — this
   is what closes the domain gap):
   ```bash
   python pipeline/convert_anchor_knnvc.py --keyword washington
   ```
5. **Build the AS-norm impostor cohort:**
   ```bash
   python pipeline/cohort_builder.py --keyword washington
   ```
6. **Calibrate the per-keyword detection threshold:**
   ```bash
   python pipeline/calibrate.py --keyword washington
   ```
7. **Run detection:**
   ```bash
   python pipeline/detector.py --keyword washington
   ```
8. **Verify candidates with the phoneme filter** (precision stage):
   ```bash
   python pipeline/verify_detections.py --keyword washington
   ```
9. **Validate against Whisper ground truth** (prints precision/recall/F1):
   ```bash
   python pipeline/validate_detection.py --keyword washington
   ```

For the full experimental history — every keyword tried, every failure mode
found, and the reasoning behind each fix — see
[Reports/EXPERIMENT_LOG.md](Reports/EXPERIMENT_LOG.md) and
[Reports/SIAMESE_PROGRESS_REPORT.md](Reports/SIAMESE_PROGRESS_REPORT.md).
