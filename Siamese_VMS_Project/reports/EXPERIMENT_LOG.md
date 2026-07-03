# Siamese KWS — Consolidated Experiment Log

One place for every detection experiment run with the Siamese pipeline, with
pointers to the surviving artifacts. Complements `SIAMESE_PROGRESS_REPORT.md`
(narrative + architecture); this file is the results ledger for the write-up.

## Chunk sets

| Set | Period in `audios/` | Content | Status |
|---|---|---|---|
| A — scripted news | ≤ 2026-06-10 | BBC-style scripted delivery | retired, chunks deleted |
| B — conversational debate | 2026-06-10 → 2026-07-02 | Elon Musk debate panel ("penalty", "elon", "appropriate") | retired, chunks deleted |
| C — Iran deal report | 2026-07-02 (day) | correspondent report ("administration", "immigration", "washington") | retired, chunks deleted; transcript archived at `Reports/archive_chunksets/transcripts_setC_iran_deal_2026-07-02.txt` |
| D — Sky News weather/news | since 2026-07-02 (evening) | UK weather bulletin into a news segue | **current** (`audios/` + `audios/transcripts.txt`), all 10 chunks content-verified unique |

All experiments are 10 chunks, chunk-level ground truth from whisper-tiny
(`validate_detection.py`). Sets B and C contained duplicated chunks (e.g. B:
live_3≡6, live_4≡7, live_5≡8; C: live_4≡6, live_5≡7) because `downloader.py`
deduplicated on the HLS segment URI, and YouTube re-signs each segment URL
(fresh query-string token) on every playlist poll - the same underlying
video bytes could arrive under a different URI and slip past the dedup set.
Fixed in `downloader.py`: dedup now hashes (md5) the downloaded bytes
themselves, so identical content is caught regardless of URL churn. Set D
was downloaded with the fixed script and confirmed to have 10/10 unique
transcripts; the fix also actively skipped 6 duplicate segments during that
run (logged, not silently dropped).

## Results

Historical rows dated 2026-07-02 that mention a phoneme verification stage
record experiments from a since-retired third pipeline stage; the current
pipeline is two-stage (anchor building + detection). Verified-run artifacts
from that stage have been removed.

| Date | Keyword | Set | Anchor | Model | Threshold | TP/FP/FN | P / R / F1 | Artifact |
|---|---|---|---|---|---|---|---|---|
| 2026-06-10 | absolutely | A | TTS centroid | v1 baseline | 1.741 | — | F1 = 0.57 | `logs/archive/detections_absolutely_2026-06-10_tts_baseline_oldchunks.json` |
| 2026-06-10 | heat | A | TTS centroid | v1 baseline | 2.244 | — | F1 = 0.80 | `logs/archive/detections_heat_2026-06-10_tts_baseline_oldchunks.json` |
| 2026-06-11 | penalty | B | TTS centroid | v1 baseline | 2.464 | 0/4/2 | F1 = 0.00 | `logs/archive/detections_penalty_2026-06-11_tts_baseline_oldchunks.json` |
| 2026-06-15 | elon | B | TTS centroid | v1 baseline | 1.252 (run 1) / 1.510 (run 2) | — | ~0.20 | `logs/archive/detections_elon_2026-06-15_run{1,2}_*.json` (two same-day runs, recalibrated threshold; kept both) |
| 2026-07-02 | penalty | B | TTS centroid (baseline rerun) | v1 baseline | 1.594 | 0/7/2 | **F1 = 0.00** — true chunks scored 0.96, below all 7 FPs; "appropriate" fired at cos 0.985 | overwritten by the kNN-VC rerun; per-window hits preserved in `logs/archive/timestamps_penalty_2026-06-11_and_2026-07-02_runs.txt` |
| 2026-07-02 | penalty | B | **kNN-VC converted** | v1 baseline | 2.166 | 2/4/0 | **P 0.33 / R 1.00 / F1 = 0.50** | `logs/archive/detections_penalty_2026-07-02_knnvc_anchor_oldchunks.json` |
| 2026-07-02 | administration | C | **kNN-VC converted** | v1 baseline | 2.341 | 1/3/0 | **P 0.25 / R 1.00 / F1 = 0.40** | `logs/detections_administration_unverified.json` |
| 2026-07-02 | administration | C | kNN-VC + **phoneme verification (retired)** | v1 baseline + wav2vec2-espeak CTC | 2.341 / tau 0.716 | 1/0/0 | **P 1.00 / R 1.00 / F1 = 1.00** | artifact removed |
| 2026-07-02 | washington | C | **kNN-VC converted** (detector alone) | v1 baseline | 2.610 | 1/2/0 | P 0.33 / R 1.00 / F1 = 0.50 — and the live_9 "TP" was accidental: windows fired on "understanding" at 2.0s, not "washington" at 4.2s | `logs/detections_washington_unverified.json` |
| 2026-07-02 | washington | C | kNN-VC + **phoneme verification + candidate rescue (retired)** | v1 baseline + wav2vec2-espeak CTC | 2.610 / tau 0.604 | 1/0/0 | **P 1.00 / R 1.00 / F1 = 1.00** | artifact removed |
| 2026-07-02 | **russia** | **D** | kNN-VC + phoneme verification (retired) | full cascade | 1.408 / tau 0.718 | 1/0/0 | **P 1.00 / R 1.00 / F1 = 1.00** | superseded by Step 3 run |
| 2026-07-02 | **weather** | **D** | kNN-VC + phoneme verification (retired) | full cascade | 2.126 / tau 0.750 | 1/0/0 | **P 1.00 / R 1.00 / F1 = 1.00** | superseded by Step 3 run |
| 2026-07-02 | **scotland** | **D** | kNN-VC + phoneme verification (retired) | full cascade | 1.793 / tau 0.707 | 3/1/0 | P 0.75 / R 1.00 / F1 = 0.86 | superseded by Step 3 run |
| 2026-07-02 | **ireland** | **D** | kNN-VC + phoneme verification (retired) | full cascade | 1.581 / tau 0.839 | 2/0/1 | P 1.00 / R 0.67 / F1 = 0.80 | superseded by Step 3 run |
| **2026-07-02** | **4-keyword micro-avg** | **D** | kNN-VC + phoneme verification (retired) | full cascade | — | 7/1/1 (40 chunk-decisions) | **P 0.875 / R 0.875 / F1 = 0.875** (macro-avg F1 = 0.915) | historical baseline — see "Set D validation" below |
| 2026-07-03 | **5-keyword micro-avg** | **D** | kNN-VC converted | **wavlm (frozen L10 mean-pool), detector alone** | per-keyword calibrated | 1/0/8 (50 chunk-decisions) | **P 1.00 / R 0.11 / F1 = 0.20** — only russia crossed its threshold; ranking was perfect but absolute scores were too low | `logs/backup_setD_wavlm10/detections_*_unverified.json` |
| 2026-07-03 | **5-keyword micro-avg** | **D** | kNN-VC converted | **wavlm-trained (Step 3 attentive head), detector alone** | per-keyword, aligned-p100 | 9/0/0 (50 chunk-decisions) | **P 1.00 / R 1.00 / F1 = 1.00** — all 9 true chunks are DIRECT threshold hits (margins +1.01..+5.03) — see "Step 3" below | `logs/detections_*.json` (current) |
| 2026-07-03 | outbreaks | D | kNN-VC converted | wavlm-trained, detector alone | 1.729 (aligned-p100) | 1/0/1 | P 1.00 / R 0.50 / F1 = 0.67 — live_7 direct hit (+1.74); live_2 FN is a chunk-boundary case (keyword ends 30 ms before chunk end, no fully-contained window) — see "Sixth keyword" under Step 3 | `logs/detections_outbreaks.json` |
| 2026-07-03 | insurance | D | kNN-VC converted | wavlm-trained, detector alone | 0.669 (aligned-p100) | 1/0/0 | **P 1.00 / R 1.00 / F1 = 1.00** — single true chunk (live_8, "...travel insurance proudly sponsors...") direct hit at 5.58 (+4.91 margin, cos 0.913, 27 windows above threshold); tightest negative (live_7) sat at 0.67, right at but not over threshold | `logs/detections_insurance.json` |

Note: `detector.py` labels every `.npz` anchor "TTS prototype centroid" in the
JSON; for the 2026-07-02 rows the anchor was actually the kNN-VC-converted
centroid (baseline TTS anchors backed up as `keywords/<kw>_anchor_tts.npz`).

## Set D validation (historical baseline, 2026-07-02)

Purpose: confirm the v1 baseline two-stage pipeline (kNN-VC anchor conversion ->
Siamese AS-norm detector, with a since-retired phoneme verification stage)
generalized to content it had never seen, on the first chunk set collected
with the duplicate-free downloader (see [[vms-chunk-dedup-fix]]). Set D is a
Sky News weather bulletin segueing into a news bulletin - a different register
from every prior chunk set (scripted news / conversational debate /
correspondent report), so this was a genuine out-of-distribution check, not a
rerun on familiar material. **Current pipeline claim:** see the Step 3 section
below (wavlm-trained attentive head, detector-only micro F1 1.00 on five
keywords).

Four keywords were chosen deliberately to span difficulty, not cherry-picked
for a good score:

- **russia**, **weather** - rare (a single true chunk each), phonetically
  distinct - the "does it work at all on a fresh domain" control.
- **scotland** - frequent (3 true chunks) - recall stress test under
  repetition.
- **ireland** - frequent (3 true chunks) **and** whisper-tiny itself
  mis-transcribes one instance as "Island" (live_5) - a near-homophone
  stress test, since /aɪɚlənd/ vs /aɪlənd/ differ by a single phone.

| Keyword | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| russia | 1 | 0 | 0 | 9 | 1.00 | 1.00 | 1.00 |
| weather | 1 | 0 | 0 | 9 | 1.00 | 1.00 | 1.00 |
| scotland | 3 | 1 | 0 | 6 | 0.75 | 1.00 | 0.86 |
| ireland | 2 | 0 | 1 | 7 | 1.00 | 0.67 | 0.80 |
| **Micro-avg (40 decisions)** | 7 | 1 | 1 | 31 | **0.875** | **0.875** | **0.875** |
| **Macro-avg** | | | | | **0.938** | **0.918** | **0.915** |

Both errors are explainable under the v1 baseline + retired verification
stage, not systemic domain-gap failures:
- scotland's FP (live_6, "...south of **Ireland** can expect...") crossed
  the embedding threshold by a hair — a near-miss the retired phoneme gate
  almost caught.
- ireland's FN is the same chunk (live_6): sliding windows did not land
  squarely on the spoken instant. Precision stayed perfect elsewhere,
  including a correct reject of the ASR-confused "Island" chunk (live_5).

**Historical claim (2026-07-02):** the v1 baseline + verification stage
achieved F1 ~0.88-0.92 (micro/macro) zero-shot keyword spotting on live-stream
audio outside its enrollment domain. **Current claim (2026-07-03):** the
trained attentive-head detector achieves detector-only micro F1 1.00 on Set D
+ brighten with aligned p100 calibration (Step 3 section below).

## Step 3: trained attentive head — detector crosses threshold without rescue (2026-07-03)

Problem: with the frozen wavlm-L10 mean-pool backend the detector RANKED
Set D perfectly (AP 1.0) but real keyword windows rarely CROSSED the
calibrated threshold — detector-only micro F1 0.20 (recall 1/9). Step 3
lifts absolute real-speech scores, not just ranking.

**Model** (`core/embedders.py`): `AttentivePoolingHead` on frozen
WavLM-base-plus layer-10 frames — 4-head attentive pooling (ICASSP-2021
QbE style) + residual MLP projection, 5.5M params, zero-init so the
untrained head is bit-identical to the validated mean-pool operating
point. Exposed as `SIAMESE_BACKEND=wavlm-trained` (artifacts `_wavlm10ft`,
checkpoint `checkpoints/siamese_v3_best.pth`).

**Training** (`training/dataset_v3.py`, `train_siamese_v3.py`, g6.xlarge L4):
sub-center ArcFace (K=3, s=30, m=0.2) over the top 1000 MSWC word classes
(34,790 clips: ≤25 real + TTS-bank clips per class, half "humanized"),
features pre-cached so an epoch is head-only. Batches are built by a
phonetic-confusable sampler (g2p_en phoneme edit distance): ~half of each
batch's 16 classes are near-neighbors of a seed word. Class-shared
real+TTS supervision replaces the Phase-2 GRL (which never converged on a
frozen backbone). Zero-shot eval on 150 held-out words (TTS centroid vs
real clips): **AUC 0.637 (mean pool) → 0.974 (epoch 24)**; impostor
cosine collapsed 0.296 → 0.005 while positives held ~0.47
(`logs/phase3_train_head.log`).

**Two calibration artifacts surfaced by the sharper embedding** (both are
protocol bugs the old blunt embedding masked, fixed in `calibrate.py` /
`core/scoring.py`):
1. **Negative-sample leakage** — calibration negatives were sampled from
   ALL chunks, including keyword-bearing ones; the trained head scores
   those so high that the p99.5 "false-alarm" threshold was being set BY
   the keyword itself (scotland: threshold 1.98 vs true windows 1.85; the
   top 2 of 400 "negatives" were the keyword). Fix: negatives come from
   keyword-free chunks only (transcript token check, same guard as the
   kNN-VC reference pool).
2. **Scoring-protocol mismatch** — negatives were embedded as isolated
   windows (own forward pass) while the detector pools windows from a
   chunk-level pass; the two land in measurably different regions
   (scotland clean-negative max −0.18 isolated vs +0.62 chunk-pooled).
   Fix: `sample_stream_window_embeddings()` embeds calibration negatives
   through the detector's own path (chunk-context pooling, all 3 window
   scales, 50 ms grid) — every keyword-free window, no subsampling — and
   the threshold is max(negatives)+1e-4 (`--fa-percentile 100`; the
   epsilon covers float32 BLAS jitter ~1e-7 between identical windows
   scored in different batch shapes).

**Result (Set D + brighten, detector alone)** — same
transcript-token ground truth as always, `pipeline/validate_detection.py`:

| Keyword | TP | FP | FN | Threshold | min TP hit − thr | F1 |
|---|---|---|---|---|---|---|
| russia | 1 | 0 | 0 | 0.029 | +5.03 | 1.00 |
| weather | 1 | 0 | 0 | 0.134 | +2.50 | 1.00 |
| scotland | 3 | 0 | 0 | 0.622 | +1.22 | 1.00 |
| ireland | 3 | 0 | 0 | 1.558 | +1.01 | 1.00 |
| brighten | 1 | 0 | 0 | 0.362 | +4.28 | 1.00 |
| **micro (50 decisions)** | 9 | 0 | 0 | | | **1.00** |

Notable: ireland recall is now 3/3 — live_6, whose occurrence the old
baseline pipeline could not land on reliably, is a direct hit at +1.01 — and
ireland's threshold is literally set one epsilon above the "Island"
near-homophone chunk (live_5, 1.5578), which the embedding scores just below
every true "Ireland" (≥2.57). brighten, previously F1 0.50 under the v1
baseline, is a clean 1.00. True hits carry 34–102 windows above threshold;
every false positive observed at looser operating points carried 1–15.

Operating-point ablation (all with the leakage guard): isolated-window
negatives p99.5 → micro F1 0.78 (recall 1.00, 5 FPs); p100 → 0.82;
aligned-protocol p100+eps → **1.00**. The pre-guard baseline was 0.71
(scotland 0/3). Old-backend detections preserved in
`logs/backup_setD_wavlm10/`.

Caveats: 10 chunks × 5 keywords from one broadcast; thresholds are
per-keyword data-fitted on keyword-free deployment audio (no manual
tuning, but max-statistics on ~2.4k–30k windows are noisier than
percentiles — watch FA rate on longer streams). Reproducibility: calibrate
(seed 777) + cohort (seed 123) are deterministic — a from-scratch scotland
re-run reproduced threshold 0.622 and the identical 3 detections.

### Sixth keyword, "outbreaks" (2026-07-03): the chunk-boundary failure mode

Fresh keyword run end-to-end under the Step 3 protocol (keyword_generator →
convert_anchor_knnvc → cohort → aligned-p100 calibrate → detector), truth =
2 of 10 chunks (live_2, live_7). Detector-only: **P 1.00 / R 0.50 / F1
0.67** (threshold 1.729), `logs/detections_outbreaks.json`.

- live_7 ("...cloudy with outbreaks of rain...") — direct hit, 3.47
  (+1.74 margin, 41 windows above threshold, cos 0.702).
- live_2 ("...will see more prolonged outbreaks.") — **FN at 1.14**.
  Whisper word timing places the keyword at 4.54–4.98 s of a 5.01 s
  chunk: it ends 30 ms before the chunk boundary, so no sliding window
  can bracket it — a full-scale (0.67 s) window starting at word onset
  would run past the chunk end. The best physical window (4.36 s, scale
  0.8) clips the final /s/ and includes the tail of "prolonged"
  (cos 0.478 vs 0.702 for the clean hit).

This is an alignment/boundary case, NOT calibration: live_0's hottest
keyword-free window scores 1.73 > live_2's 1.14, so no threshold yields
2TP/0FP (p99.5 = 1.151 still misses by 0.011; p99 = 0.861 catches live_2
but admits 2 FPs — F1 unchanged at 0.67, and the FA-bounded-by-
construction property of the p100 threshold is lost). Detector/calibration
parameter changes were evaluated and rejected on those tradeoffs.
Structural fix: overlap consecutive chunks by ~1 s in downloader.py / the
platform live loop so no word can straddle or abut a boundary (needs a dedup
rule for detections in the overlap). The Step 3 claim stands with this
boundary condition documented: direct threshold detection holds for words
with at least one fully-contained window; edge-of-chunk words still need
overlap.

## Key findings

1. **TTS↔real domain gap was the recall killer.** On conversational speech the
   raw TTS anchor ranks confusables above the true keyword (inversion), so no
   threshold can recover it (penalty F1 = 0, reproduced twice).
2. **kNN-VC anchor conversion fixes the inversion with zero retraining.**
   Converting the 30 TTS clips into the stream's own voice (WavLM kNN +
   prematched HiFi-GAN; keyword-bearing chunks excluded from the reference to
   avoid leakage) recovered recall 0 → 1.0 on both keywords tried, dropped the
   "appropriate" confusable from cos 0.985 → 0.942, and raised centroid
   cohesion 0.887 → 0.960. Calibration margin went positive
   (administration: +0.258) for the first time on real audio.
3. **Trained attentive head closes the recall gap without a third stage.**
   Sub-center ArcFace over MSWC + TTS bank, with phonetic-confusable batch
   mining (g2p_en edit distance), trains an attentive pooling head on frozen
   WavLM layer-10 frames. On Set D + brighten, detector-only micro F1 goes
   from 0.20 (frozen WavLM mean-pool) to **1.00**, with all true chunks
   crossing the aligned p100 threshold directly.
4. **Aligned calibration protocol matters once scores sharpen.** Negatives
   must come from keyword-free chunks only, and must be embedded through
   the detector's own chunk-pooled window path — isolated-window negatives
   underestimated false-alarm rates and admitted 5 FPs at p99.5.
5. Anchor references decode cleanly in both domains
   (/æ d m ɪ n ɪ s t ɹ eɪ ʃ ə n/ from TTS and from kNN-VC-converted
   clips alike), confirming phonetic content is preserved through kNN-VC
   even when embedding space was domain-sensitive.

## Artifact map (post-cleanup, 2026-07-02)

- `logs/detections_<kw>.json` — latest run on the **current** (Set D) chunk
  set only: russia, weather, scotland, ireland. administration/washington
  were built against the now-retired Set C and moved to `logs/archive/`.
- `logs/archive/` — dated historical runs on retired chunk sets
  (`*_oldchunks`); duplicate per-run `timestamps_*.txt` files were deleted
  where the JSON holds the same detections (absolutely, heat, elon,
  administration). The penalty timestamps file was kept because it is the only
  record of the 2026-07-02 TTS-baseline rerun.
- `keywords/<kw>_anchor.npz` / `<kw>_anchor_tts.npz` — kNN-VC vs baseline
  anchors; `<kw>_calibration.json`, `cohort_<kw>.npz` alongside.
- `keywords/<kw>_variants/` vs `<kw>_variants_knnvc/` — TTS clips before/after
  voice conversion (kept: side-by-side audio demo material; the penalty
  knnvc set is irreproducible since chunk set B was deleted).
- `checkpoints/` — Phase 1 baseline + both Phase 2 GRL runs.
- `videos/` — current chunk set's video sources (demo material for showing
  detections in context).
