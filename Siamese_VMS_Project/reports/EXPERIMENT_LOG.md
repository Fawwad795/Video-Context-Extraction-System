# Siamese KWS - Consolidated Experiment Log

One place for every detection experiment run with the Siamese pipeline, with
pointers to the surviving artifacts. Complements `SIAMESE_PROGRESS_REPORT.md`
(narrative + architecture); this file is the results ledger for the write-up.

## Chunk sets

| Set | Period in `audios/` | Content | Status |
|---|---|---|---|
| A - scripted news | ≤ 2026-06-10 | BBC-style scripted delivery | retired, chunks deleted |
| B - conversational debate | 2026-06-10 → 2026-07-02 | Elon Musk debate panel ("penalty", "elon", "appropriate") | retired, chunks deleted |
| C - Iran deal report | 2026-07-02 (day) | correspondent report ("administration", "immigration", "washington") | retired, chunks deleted; transcript archived at `Reports/archive_chunksets/transcripts_setC_iran_deal_2026-07-02.txt` |
| D - Sky News weather/news | since 2026-07-02 (evening) | UK weather bulletin into a news segue | **current** (`audios/Chunkset_D/`), all 10 chunks content-verified unique |
| E - Iran deal report | added 2026-07-04 | correspondent report ("administration", "immigration") - same broadcast register as retired Set C but a fresh chunk pull | **current** (`audios/Chunkset_E/`); contains duplicated pairs (live_4≡6, live_5≡7) |
| F - US Independence Day feature | 2026-07-04 | Sky News package on the US anniversary celebrations (Trump, state fair, vox pops) - **20 chunks**, first set larger than 10 | **current** (`audios/Chunkset_F/`), all 20 content-verified unique |

Since 2026-07-04 chunk sets live side by side under `audios/<Chunkset>/`
(each with its own `transcripts.txt`); select one via `SIAMESE_AUDIO_DIR`.

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
| 2026-06-10 | absolutely | A | TTS centroid | v1 baseline | 1.741 | - | F1 = 0.57 | `logs/archive/detections_absolutely_2026-06-10_tts_baseline_oldchunks.json` |
| 2026-06-10 | heat | A | TTS centroid | v1 baseline | 2.244 | - | F1 = 0.80 | `logs/archive/detections_heat_2026-06-10_tts_baseline_oldchunks.json` |
| 2026-06-11 | penalty | B | TTS centroid | v1 baseline | 2.464 | 0/4/2 | F1 = 0.00 | `logs/archive/detections_penalty_2026-06-11_tts_baseline_oldchunks.json` |
| 2026-06-15 | elon | B | TTS centroid | v1 baseline | 1.252 (run 1) / 1.510 (run 2) | - | ~0.20 | `logs/archive/detections_elon_2026-06-15_run{1,2}_*.json` (two same-day runs, recalibrated threshold; kept both) |
| 2026-07-02 | penalty | B | TTS centroid (baseline rerun) | v1 baseline | 1.594 | 0/7/2 | **F1 = 0.00** - true chunks scored 0.96, below all 7 FPs; "appropriate" fired at cos 0.985 | overwritten by the kNN-VC rerun; per-window hits preserved in `logs/archive/timestamps_penalty_2026-06-11_and_2026-07-02_runs.txt` |
| 2026-07-02 | penalty | B | **kNN-VC converted** | v1 baseline | 2.166 | 2/4/0 | **P 0.33 / R 1.00 / F1 = 0.50** | `logs/archive/detections_penalty_2026-07-02_knnvc_anchor_oldchunks.json` |
| 2026-07-02 | administration | C | **kNN-VC converted** | v1 baseline | 2.341 | 1/3/0 | **P 0.25 / R 1.00 / F1 = 0.40** | `logs/detections_administration_unverified.json` |
| 2026-07-02 | administration | C | kNN-VC + **phoneme verification (retired)** | v1 baseline + wav2vec2-espeak CTC | 2.341 / tau 0.716 | 1/0/0 | **P 1.00 / R 1.00 / F1 = 1.00** | artifact removed |
| 2026-07-02 | washington | C | **kNN-VC converted** (detector alone) | v1 baseline | 2.610 | 1/2/0 | P 0.33 / R 1.00 / F1 = 0.50 - and the live_9 "TP" was accidental: windows fired on "understanding" at 2.0s, not "washington" at 4.2s | `logs/detections_washington_unverified.json` |
| 2026-07-02 | washington | C | kNN-VC + **phoneme verification + candidate rescue (retired)** | v1 baseline + wav2vec2-espeak CTC | 2.610 / tau 0.604 | 1/0/0 | **P 1.00 / R 1.00 / F1 = 1.00** | artifact removed |
| 2026-07-02 | **russia** | **D** | kNN-VC + phoneme verification (retired) | full cascade | 1.408 / tau 0.718 | 1/0/0 | **P 1.00 / R 1.00 / F1 = 1.00** | superseded by Step 3 run |
| 2026-07-02 | **weather** | **D** | kNN-VC + phoneme verification (retired) | full cascade | 2.126 / tau 0.750 | 1/0/0 | **P 1.00 / R 1.00 / F1 = 1.00** | superseded by Step 3 run |
| 2026-07-02 | **scotland** | **D** | kNN-VC + phoneme verification (retired) | full cascade | 1.793 / tau 0.707 | 3/1/0 | P 0.75 / R 1.00 / F1 = 0.86 | superseded by Step 3 run |
| 2026-07-02 | **ireland** | **D** | kNN-VC + phoneme verification (retired) | full cascade | 1.581 / tau 0.839 | 2/0/1 | P 1.00 / R 0.67 / F1 = 0.80 | superseded by Step 3 run |
| **2026-07-02** | **4-keyword micro-avg** | **D** | kNN-VC + phoneme verification (retired) | full cascade | - | 7/1/1 (40 chunk-decisions) | **P 0.875 / R 0.875 / F1 = 0.875** (macro-avg F1 = 0.915) | historical baseline - see "Set D validation" below |
| 2026-07-03 | **5-keyword micro-avg** | **D** | kNN-VC converted | **wavlm (frozen L10 mean-pool), detector alone** | per-keyword calibrated | 1/0/8 (50 chunk-decisions) | **P 1.00 / R 0.11 / F1 = 0.20** - only russia crossed its threshold; ranking was perfect but absolute scores were too low | `logs/backup_setD_wavlm10/detections_*_unverified.json` |
| 2026-07-03 | **5-keyword micro-avg** | **D** | kNN-VC converted | **wavlm-trained (Step 3 attentive head), detector alone** | per-keyword, aligned-p100 | 9/0/0 (50 chunk-decisions) | **P 1.00 / R 1.00 / F1 = 1.00** - all 9 true chunks are DIRECT threshold hits (margins +1.01..+5.03) - see "Step 3" below | `logs/detections_*.json` (current) |
| 2026-07-03 | outbreaks | D | kNN-VC converted | wavlm-trained, detector alone | 1.729 (aligned-p100) | 1/0/1 | P 1.00 / R 0.50 / F1 = 0.67 - live_7 direct hit (+1.74); live_2 FN is a chunk-boundary case (keyword ends 30 ms before chunk end, no fully-contained window) - see "Sixth keyword" under Step 3 | `logs/detections_outbreaks.json` |
| 2026-07-03 | insurance | D | kNN-VC converted | wavlm-trained, detector alone | 0.669 (aligned-p100) | 1/0/0 | **P 1.00 / R 1.00 / F1 = 1.00** - single true chunk (live_8, "...travel insurance proudly sponsors...") direct hit at 5.58 (+4.91 margin, cos 0.913, 27 windows above threshold); tightest negative (live_7) sat at 0.67, right at but not over threshold | `logs/detections_insurance.json` |
| 2026-07-04 | western | D | kNN-VC converted | wavlm-trained, detector alone, combined_best defaults | 6.228 (aligned-p100) | 1/0/0 | **P 1.00 / R 1.00 / F1 = 1.00** - first run with the ablation sweet-spot defaults baked in; pipeline 2.7 min end-to-end | `logs/detections_western.json` |
| 2026-07-04 | sunny / windy / showers | D | **TTS centroid AND kNN-VC converted (both)** | wavlm-trained, detector alone | per-keyword aligned-p100 | 3/0/0 each way | **F1 = 1.00 both with and without kNN-VC, all 3 keywords** (kNN-VC retirement ablation) | `ablation_study/results/knnvc_ablation.jsonl` |
| 2026-07-04 | administration | **E** | **TTS centroid (no kNN-VC)** | wavlm-trained, detector alone | 3.276 (aligned-p100) | 1/0/0 | **P 1.00 / R 1.00 / F1 = 1.00** - near-homophone "immigration" chunk rejected at 2.04; compare F1 0.40 for the same keyword on Set C under v1 baseline + kNN-VC | `logs/detections_administration.json` |

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
  the embedding threshold by a hair - a near-miss the retired phoneme gate
  almost caught.
- ireland's FN is the same chunk (live_6): sliding windows did not land
  squarely on the spoken instant. Precision stayed perfect elsewhere,
  including a correct reject of the ASR-confused "Island" chunk (live_5).

**Historical claim (2026-07-02):** the v1 baseline + verification stage
achieved F1 ~0.88-0.92 (micro/macro) zero-shot keyword spotting on live-stream
audio outside its enrollment domain. **Current claim (2026-07-03):** the
trained attentive-head detector achieves detector-only micro F1 1.00 on Set D
+ brighten with aligned p100 calibration (Step 3 section below).

## Step 3: trained attentive head - detector crosses threshold without rescue (2026-07-03)

Problem: with the frozen wavlm-L10 mean-pool backend the detector RANKED
Set D perfectly (AP 1.0) but real keyword windows rarely CROSSED the
calibrated threshold - detector-only micro F1 0.20 (recall 1/9). Step 3
lifts absolute real-speech scores, not just ranking.

**Model** (`core/embedders.py`): `AttentivePoolingHead` on frozen
WavLM-base-plus layer-10 frames - 4-head attentive pooling (ICASSP-2021
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
1. **Negative-sample leakage** - calibration negatives were sampled from
   ALL chunks, including keyword-bearing ones; the trained head scores
   those so high that the p99.5 "false-alarm" threshold was being set BY
   the keyword itself (scotland: threshold 1.98 vs true windows 1.85; the
   top 2 of 400 "negatives" were the keyword). Fix: negatives come from
   keyword-free chunks only (transcript token check, same guard as the
   kNN-VC reference pool).
2. **Scoring-protocol mismatch** - negatives were embedded as isolated
   windows (own forward pass) while the detector pools windows from a
   chunk-level pass; the two land in measurably different regions
   (scotland clean-negative max −0.18 isolated vs +0.62 chunk-pooled).
   Fix: `sample_stream_window_embeddings()` embeds calibration negatives
   through the detector's own path (chunk-context pooling, all 3 window
   scales, 50 ms grid) - every keyword-free window, no subsampling - and
   the threshold is max(negatives)+1e-4 (`--fa-percentile 100`; the
   epsilon covers float32 BLAS jitter ~1e-7 between identical windows
   scored in different batch shapes).

**Result (Set D + brighten, detector alone)** - same
transcript-token ground truth as always, `pipeline/validate_detection.py`:

| Keyword | TP | FP | FN | Threshold | min TP hit − thr | F1 |
|---|---|---|---|---|---|---|
| russia | 1 | 0 | 0 | 0.029 | +5.03 | 1.00 |
| weather | 1 | 0 | 0 | 0.134 | +2.50 | 1.00 |
| scotland | 3 | 0 | 0 | 0.622 | +1.22 | 1.00 |
| ireland | 3 | 0 | 0 | 1.558 | +1.01 | 1.00 |
| brighten | 1 | 0 | 0 | 0.362 | +4.28 | 1.00 |
| **micro (50 decisions)** | 9 | 0 | 0 | | | **1.00** |

Notable: ireland recall is now 3/3 - live_6, whose occurrence the old
baseline pipeline could not land on reliably, is a direct hit at +1.01 - and
ireland's threshold is literally set one epsilon above the "Island"
near-homophone chunk (live_5, 1.5578), which the embedding scores just below
every true "Ireland" (≥2.57). brighten, previously F1 0.50 under the v1
baseline, is a clean 1.00. True hits carry 34-102 windows above threshold;
every false positive observed at looser operating points carried 1-15.

Operating-point ablation (all with the leakage guard): isolated-window
negatives p99.5 → micro F1 0.78 (recall 1.00, 5 FPs); p100 → 0.82;
aligned-protocol p100+eps → **1.00**. The pre-guard baseline was 0.71
(scotland 0/3). Old-backend detections preserved in
`logs/backup_setD_wavlm10/`.

Caveats: 10 chunks × 5 keywords from one broadcast; thresholds are
per-keyword data-fitted on keyword-free deployment audio (no manual
tuning, but max-statistics on ~2.4k-30k windows are noisier than
percentiles - watch FA rate on longer streams). Reproducibility: calibrate
(seed 777) + cohort (seed 123) are deterministic - a from-scratch scotland
re-run reproduced threshold 0.622 and the identical 3 detections.

### Sixth keyword, "outbreaks" (2026-07-03): the chunk-boundary failure mode

Fresh keyword run end-to-end under the Step 3 protocol (keyword_generator →
convert_anchor_knnvc → cohort → aligned-p100 calibrate → detector), truth =
2 of 10 chunks (live_2, live_7). Detector-only: **P 1.00 / R 0.50 / F1
0.67** (threshold 1.729), `logs/detections_outbreaks.json`.

- live_7 ("...cloudy with outbreaks of rain...") - direct hit, 3.47
  (+1.74 margin, 41 windows above threshold, cos 0.702).
- live_2 ("...will see more prolonged outbreaks.") - **FN at 1.14**.
  Whisper word timing places the keyword at 4.54-4.98 s of a 5.01 s
  chunk: it ends 30 ms before the chunk boundary, so no sliding window
  can bracket it - a full-scale (0.67 s) window starting at word onset
  would run past the chunk end. The best physical window (4.36 s, scale
  0.8) clips the final /s/ and includes the tail of "prolonged"
  (cos 0.478 vs 0.702 for the clean hit).

This is an alignment/boundary case, NOT calibration: live_0's hottest
keyword-free window scores 1.73 > live_2's 1.14, so no threshold yields
2TP/0FP (p99.5 = 1.151 still misses by 0.011; p99 = 0.861 catches live_2
but admits 2 FPs - F1 unchanged at 0.67, and the FA-bounded-by-
construction property of the p100 threshold is lost). Detector/calibration
parameter changes were evaluated and rejected on those tradeoffs.
Structural fix: overlap consecutive chunks by ~1 s in downloader.py / the
platform live loop so no word can straddle or abut a boundary (needs a dedup
rule for detections in the overlap). The Step 3 claim stands with this
boundary condition documented: direct threshold detection holds for words
with at least one fully-contained window; edge-of-chunk words still need
overlap.

## kNN-VC retirement (2026-07-04)

kNN-VC anchor conversion (finding 2 below) was built to fix the TTS↔real
domain gap under the v1 frozen-backbone baseline. Two lines of evidence
showed the Step-3 trained head makes it redundant:

1. **Accidental ablation.** A defaults-sync bug (convert_anchor_knnvc.py
   `--holdout` 6 vs keyword_generator.py's new 4) made the conversion
   silently no-op (bare `return`, exit 0) for the `voices_7`/`combined_best`
   pipeline-parameter ablation configs - which still scored micro F1 1.00
   on all 5 Set D keywords with the raw TTS anchor.
2. **Deliberate ablation** (`ablation_study/knnvc_ablation.py`, archived
   branch): with the bug fixed, 3 fresh Set D keywords (sunny, windy,
   showers) run both ways - F1 1.00 identical, kNN-VC only adding ~50 s
   per keyword. Then `administration` on the new Set E without kNN-VC:
   F1 1.00 with the near-homophone "immigration" rejected (2.04 vs
   threshold 3.28) - the confusable-pair failure mode that motivated
   kNN-VC in the first place (penalty/appropriate, F1 0.00, finding 1).

9/9 keywords across two chunk sets show no F1 benefit. The stage was
removed from the pipeline, platform worker, and ablation harness on
2026-07-04; the full implementation (convert_anchor_knnvc.py,
knnvc_ablation.py) is preserved on the **`archive/knnvc-pipeline`** branch.
Historical rows above that say "kNN-VC converted" record runs made while
the stage was active.

## Calibration leakage without transcripts (2026-07-05/06)

**Bug.** The live platform's bootstrap calibration (`calibrate.py`, ~10
chunks, no transcripts) can sample a chunk containing the keyword itself
as a "negative." Since fa_percentile=100 sets the threshold to the
highest-scoring negative window + ε, the threshold gets set to that
utterance's own score - a guaranteed miss. Observed across four keywords
on the live platform, each missing the exact utterance that contaminated
its own calibration by one epsilon: `south` (4.3851 vs threshold 4.3852),
`brain` (5.74 vs 5.743), `morning` (5.46 vs 5.463), `mexico` (7.24 vs
7.244).

**A full day (2026-07-05) was spent on transcription-free fixes, all
tried, all reverted (commit `181ea7f`, restoring `core/scoring.py`,
`pipeline/calibrate.py`, `pipeline/cohort_builder.py`,
`platform/live_worker.py` to their pre-saga state at `706a275`):**

1. *Score-in-positive-range excision* - a percentile-based reference from
   the ~12 TTS positives was too fragile at that sample size (jumped
   either too strict, missing real leaks, or too loose, misidentifying
   genuine hard negatives as leaks and regressing leak-free calibration).
2. *Temporal-burst clustering* - sliding-window overlap means ANY
   strongly-matching moment, keyword or confusable, produces the same
   clustering signature; not discriminative, also regressed clean data.
3. *Generalized Pareto (EVT) tail fit* (`scoring.evt_threshold`, kept as
   an opt-in research flag, not the default) - the most informative
   negative result: a rigorous tail fit built from hundreds of points, not
   one heuristic, still judged the leaked score statistically
   unremarkable. No per-sample statistic can tell a leak apart from an
   equally-extreme confusable, because that overlap is *why* p100 rather
   than a percentile is the right rule to begin with.
4. *Fixed-tolerance periodic recalibration* - excluding a small fixed
   count of a growing pool's top scores before taking the max forces the
   threshold below scores already known to be negatives once the pool is
   small (a live `morning` session: threshold 5.463 → 2.065 on 6 pooled
   chunks, ~7 clean chunks then fired as false positives - a sweep).
5. *Held-chunk leave-one-out adjudication* - held no-match chunks until
   real bootstrap-scale evidence existed (30 chunks), then ran one
   adjudication: the highest-scoring held chunk fires only if it beats
   every other held chunk AND sits at the current threshold's epsilon. No
   tolerance, so no sweep - but bounded to recovering **at most one**
   contaminated chunk per session by construction. A live `mexico` session
   (a Mexico-focused broadcast, where the word is common rather than rare)
   broke this differently: at least 6 genuine utterances were present
   among the first 30 held chunks; only 1 was recovered, the other 5+
   were permanently deleted in that same pass, and the recalibrated
   threshold was itself set by a second real utterance - so any future
   utterance at or below that new ceiling kept missing.

**Root cause of all five failures:** none of them use any information
beyond the score distribution itself, and a leaked keyword's score is
provably indistinguishable from a legitimate hard negative's using score
statistics alone (confirmed directly by finding 3 above) - worse, when
the keyword recurs often, "at most one recoverable" is a hard ceiling no
tuning of any of these designs can lift.

**The fix: restore what the offline research pipeline always relied on.**
`calibrate.py`'s `keyword_free_chunks()` transcript guard already existed
and needs no code change - it was a no-op on the platform only because no
transcript ever existed there. `platform/live_worker.py` now runs
`transcribe_chunks.py --limit <bootstrap-chunks>` once, right after the
bootstrap wait and before cohort/calibration, writing
`<data_root>/audios/transcripts.txt` so `keyword_free_chunks()` picks it
up automatically. Default model upgraded from whisper-tiny to
**whisper-base** for this call specifically (already cached locally, ~2x
tiny's params): a missed word here silently reintroduces the exact leak
this exists to prevent, so accuracy matters more than for ground-truth
evaluation elsewhere, where a little transcription noise averages out.
This is the approach originally proposed when the `south` bug first
surfaced and explicitly deferred at the time in favor of exploring
transcription-free alternatives (all five above) - reconsidered once
`mexico` proved none of them generalize to a keyword that recurs often.

**Validation (real audio + model, not synthetic scores).** The
`mexico` session's platform data was gone (cleared), so the fix was
validated by simulating a fresh platform bootstrap against Chunkset D's
10 chunks with no `transcripts.txt` present (copied without it, mirroring
a raw platform download) - the same multi-occurrence stress case
`mexico` exposed, since `scotland` occurs 3 times in that set:
1. `keyword_generator.py --keyword scotland` → anchor built normally.
2. `transcribe_chunks.py --limit 10` (whisper-base) → all 3 "Scotland"
   mentions caught correctly, matching the curated ground-truth
   transcript exactly.
3. `cohort_builder.py` + `calibrate.py --keyword scotland` → threshold
   2.315, n_neg=2316 (7 keyword-free chunks' windows, correctly excluding
   all 3 scotland-bearing chunks at once - not just the highest-scoring
   one), margin +1.452 (healthy, no domain-gap warning).
4. `detector.py` + `validate_detection.py` → **TP=3 FP=0, F1=1.00** - all
   3 occurrences detected, matching the pipeline's historical
   oracle-calibration result on this keyword exactly.

Unlike every reverted approach, this doesn't degrade as the keyword
recurs more often - exclusion is by content, not by order statistics on
a bounded sample.

## Phonetic confusables and the verifier resurrection (2026-07-07/08)

Three consecutive live platform sessions (keywords 'world', 'russia',
'party'; every fired chunk transcript-verified with whisper-base, English
forced, plus windowed re-transcription of each suspected false positive)
established a consistent failure signature: **true hits clear the
calibrated threshold comfortably while false positives enter within ~1
unit of it** - world 2 FP (3.93/4.09 vs thr 3.79, TPs >= 4.70), russia
4 FP (2.92-3.67 vs thr 2.65, TPs >= 4.93). The gap-based safety margin
(commit ae3dffe) was fitted to exactly that band.

The 'party' session (272 chunks, margin active, thr 4.701 = base 3.394 +
1.307) broke the pattern: 9 detections = 6 TP / 3 FP, and FP "policy"
scored **8.17 - above the real TP at 6.13**. The confusables were all
phonetic near-neighbours ("policy" x2, "publicly", "public" vs /pɑɹti/).
No threshold can separate overlapping bands, and every cheap structural
signal fails with the embedding score because it is a function of the same
embedding: the 8.17 "policy" FP had MORE windows above threshold (33 vs 7),
MORE scales (3 vs 2), and HIGHER raw cosine (0.717 vs 0.553) than the
weakest real TP. Session recall was 6/6 (three apparent misses turned out
to be chunks downloaded after the session stopped, never scanned; re-scanned
they score 8.5-9.8 and are real mentions).

**Fix: resurrect the Stage-3 phoneme verifier** (removed in df141b4 when
the trained head made it unnecessary for *recall* on Set D - the live
confusable regime is where its *precision* role was needed), with three
changes over the retired design, each forced by measured failures:

1. **Reference outlier filter.** The ksp TTS voice renders 'party' as
   /ph ɑ l i/ - a reference that would MATCH "policy". Leave-one-out
   agreement (drop refs < 0.5 vs the rest) removes it automatically
   (agreement 0.25 vs 1.00 for the other six voices).
2. **Wide-span decode instead of per-window probing.** The old
   narrow-window scheme (window + 0.08 s pad, +/-100 ms probes) clips word
   edges (TP live_44 decoded /p ɑu5 ɕ i5/ at its top window) and hands
   confusables a decode-noise lottery - "policy" hit 0.75 once across 147
   narrow decodes, and reproducibly in 17. Decoding ONE ~2.5 s span per
   detection event (detections time-clustered at 0.3 s gap) gives the CTC
   full acoustic context: every TP decodes a clean /p ɑːɹ ɾ i/ in-context
   (all 1.00), every confusable decodes as itself - /p ɔ l ə s i/,
   /p ʌ b l ɪ k/ (all 0.25). The infix matcher's free ends make the
   surrounding words cost nothing and accept derivatives (/p ɑːɹ ɾ i z/).
3. **tau calibration mirrors the decision statistic** (max over ~2 wide
   decodes of keyword-free stream audio per draw), midpoint to the refs'
   LOO self-similarity, floored at 0.6 → party tau 0.75.

End-to-end on the party session with the production code
(`core/phoneme_verify.py`, wired into `platform/live_worker.py` and
`pipeline/verify_detections.py`): **12/12 chunks correct** - 9 real
keyword chunks kept at phone-sim 1.00 (6 fired + the 3 late unscanned
ones re-run as future detections), 3 confusables rejected at 0.25.

A ground-truth correction the verifier itself surfaced: whisper-base
transcribes live_115 as "any political **policy** in the country", but
whisper-small, wav2vec2-base-960h (char CTC) and the espeak phone CTC all
hear "political **party**" - consistent with its 8.17 embedding score.
Chunk-level metrics that use a single ASR model as ground truth inherit
that model's confusions.

Cost: verifier model (~1.2 GB wav2vec2-large) loads once at setup; refs +
tau are cached per keyword (`keywords/<kw>_phone_cache.json`); at
detection time one CTC decode per event, only on chunks that fired.
`SIAMESE_PHONE_VERIFY=0` disables the platform stage.

## Rival-anchor verification replaces the phone stage (2026-07-11, branch `feature/rival-anchor-verifier`)

Motivation: keep the two-stage cascade's precision role but make the second
view an in-house mechanism (decision-time synthesized impostors) rather than
phone-sequence matching, and fix the phone stage's measured weakness (the
consistency sweep's only cascade miss: ireland live_6 clipped at phone-sim
0.80 vs tau 0.81, tau squeezed by "Island" in the keyword-free audio).

**Design** (`core/rival_verify.py`, `pipeline/rival_builder.py`,
`pipeline/rival_bank.py`; phone stage kept behind `SIAMESE_VERIFIER=phone`):
synthesize the keyword's confusables as explicit rival anchors and keep a
detection only if a full-word-scale window around it beats every armed rival
by a calibrated AS-norm margin. Every design element was forced by a measured
failure during bring-up, in order:

1. *Embedding-space rival selection* - phone edit distance never surfaces the
   live-observed confusables ("policy" is party's 845th phone neighbour,
   "public" its 31,727th); ranking a global TTS word bank (4k frequent words
   from the Whisper BPE vocabulary, one voice each) by anchor proximity finds
   policy at #6. Two-stage union: lexicon neighbours + bank neighbours.
2. *Effective-homophone exclusion* - arming "island" against ireland rejected
   all 3 real ireland chunks (margins −0.28..−0.57): broadcast /aɪɚlənd/ with
   reduced /ɚ/ IS /aɪlənd/. Single-edit pairs whose edit is schwa-family are
   excluded as out of acoustic scope (matches the phone stage's 0.80-vs-0.81
   squeeze on the same pair).
3. *Phone-substring exclusion* (schwa-normalized ≥0.7 contiguous overlap) -
   "ministration" rejected genuine "administration" windows: 0.6-scale
   detection windows legitimately cover partial words, so any rival that is a
   partial rendering of the keyword rejects the keyword itself. Same rule
   catches part/partly for party.
4. *Per-rival arming gates* - rival's own clips must lose by ≥0.15 AS-norm
   margin AND the keyword's holdout positives must beat it by ≥0.30 (p10);
   dropped pardee/pardi/pardy (D↔T voicing neighbours of party) and generic
   "-ation" words whose tight clean centroids out-compete the diffuse
   augmented anchor.
5. *AS-norm margins* (not raw cosine - anchor-construction bias) and
   *full-word alignment-search verification* (0.8/1.0× anchor window,
   ±0.25 s grid around the trigger, margin = max over grid) - verifying on
   the sub-word detection windows themselves replays failure 3 from the
   audio side: the top windows of a true "administration" are acoustically
   "-istration" and lose to "information".

**Validation so far** (real audio unless noted):
- ireland on D: 3/3 verified at margins +1.98..+2.15 (nearest armed rival
  "iceland") - including live_6, the phone stage's only cascade miss.
- administration on E: TP verified +0.45; forced-event probe on the real
  "immigration" chunk rejected at −1.58.
- party/policy regression (synthetic proxy; the live session's audio is
  gone): party in 3 unseen blended+augmented TTS voices → 3/3 KEEP
  (+0.006..+2.25); policy/publicly/public ×3 each → 9/9 REJECT
  (−0.29..−1.75), with the nearest-rival diagnostic naming "policy" itself.
- Full 14-eval rival-vs-phone head-to-head on identical fresh detector
  outputs (2026-07-12, final constants: delta = rival max + 0.2 x gap,
  overlap bar 0.65): **RAV cascade 27/28 TP, 0 FP, micro F1 0.982 = phone
  cascade 27/28, 0.982 - complementary single failures.** RAV's miss is
  the boundary-truncated outbreaks live_2 (no full-word verification
  window exists at a chunk edge; ingestion overlap is the fix); phone's
  miss is ireland live_6 (tau squeezed by "Island"). Detector-only stays
  28/28 (1.000). Two calibration-constant bugs were found and fixed by
  the sweep itself: a midpoint delta rule clipped real america hits that
  WON their contest (margins +2.1/+3.0 vs delta 3.17 - rule changed to
  rival-max + 0.2 x gap, house style), and "westward" (0.667 overlap,
  shared stressed onset + reduced tail) false-rejected western until the
  substring bar moved 0.7 -> 0.65. RAV is the default stage; phone kept
  as the ablation (`SIAMESE_VERIFIER=phone`).

Cost: no second model (the phone stage loaded ~1.2 GB wav2vec2-large);
verification is a few dot products against ~a dozen rival centroids.
One-time artifacts: global bank ~2 h CPU per backend; per-keyword rivals
~1-2 min (TTS) cached in `keywords/<kw>_rivals*.npz`.

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
   the detector's own chunk-pooled window path - isolated-window negatives
   underestimated false-alarm rates and admitted 5 FPs at p99.5.
5. Anchor references decode cleanly in both domains
   (/æ d m ɪ n ɪ s t ɹ eɪ ʃ ə n/ from TTS and from kNN-VC-converted
   clips alike), confirming phonetic content is preserved through kNN-VC
   even when embedding space was domain-sensitive.
6. **The trained head subsumes kNN-VC.** Once finding 3's head is in place,
   removing kNN-VC costs nothing on 9/9 keywords tested - including the
   confusable-pair case (administration/immigration) that finding 2's fix
   existed for. Retired 2026-07-04 (section above); the domain gap is now
   closed in embedding space rather than in audio space.
7. **A leaked-keyword calibration sample cannot be fixed without ground
   truth, and the amount of engineering doesn't change that.** Five
   transcription-free designs were tried (score-range/temporal-burst/EVT
   excision, fixed-tolerance recalibration, held-chunk adjudication) and
   all reverted - each fails a different way, but all share the same
   root cause: a leaked keyword's score is provably indistinguishable
   from a legitimate hard negative's using score statistics alone, and
   that indistinguishability gets worse, not better, the more often the
   keyword recurs. The fix was never a smarter statistic - it was
   restoring the transcript-based guard the offline pipeline always had,
   via one cheap one-time whisper-base transcription of just the
   bootstrap window.
8. **Phonetic confusables overlap the keyword in embedding space, and only
   a decorrelated view separates them.** Live 'party' monitoring fired on
   "policy" at AS-norm 8.17 - above a real keyword hit at 6.13 - and the
   confusable beat the weakest true hit on every embedding-derived signal
   (window support, scale consistency, raw cosine). The phone-sequence
   view separates them perfectly (true chunks 1.00, confusables 0.25)
   because CTC phone labels are supervised against phone identity, the
   one thing the confusable actually lacks. Two implementation details
   carry the result: decode a wide (~2.5 s) span per detection event
   rather than probing narrow windows (context stabilizes the CTC and
   removes the confusable's decode-noise lottery), and drop TTS reference
   decodes that disagree with the voice consensus (one accented voice
   rendered 'party' as /ph ɑ l i/ - a reference that would have matched
   the very confusable the verifier exists to reject).

9. **The absolute-vs-relative distinction, not the second model, is what
   separates confusables.** Finding 8 concluded the confirmation must come
   from a different information source; the rival-anchor stage shows a
   *relative* test in the SAME embedding space suffices: real confusable
   audio loses the margin contest to its own rival anchor (policy probes
   −1.1..−1.5) while true keyword audio wins at some full-word alignment.
   What actually failed in the live 'party' incident was every signal
   derived from one absolute score against one anchor. The embedding's
   resolution floor still exists - homophones and single-schwa-edit pairs
   are excluded as unseparable by construction - and the phone view keeps
   independent value there only in principle, since the same pair also sat
   at its resolution floor (0.80 vs tau 0.81).

## Artifact map (post-cleanup, 2026-07-02)

- `logs/detections_<kw>.json` - latest run on the **current** (Set D) chunk
  set only: russia, weather, scotland, ireland. administration/washington
  were built against the now-retired Set C and moved to `logs/archive/`.
- `logs/archive/` - dated historical runs on retired chunk sets
  (`*_oldchunks`); duplicate per-run `timestamps_*.txt` files were deleted
  where the JSON holds the same detections (absolutely, heat, elon,
  administration). The penalty timestamps file was kept because it is the only
  record of the 2026-07-02 TTS-baseline rerun.
- `keywords/<kw>_anchor.npz` / `<kw>_anchor_tts.npz` - kNN-VC vs baseline
  anchors; `<kw>_calibration.json`, `cohort_<kw>.npz` alongside.
- `keywords/<kw>_variants/` vs `<kw>_variants_knnvc/` - TTS clips before/after
  voice conversion (kept: side-by-side audio demo material; the penalty
  knnvc set is irreproducible since chunk set B was deleted).
- `checkpoints/` - Phase 1 baseline + both Phase 2 GRL runs.
- `videos/` - current chunk set's video sources (demo material for showing
  detections in context).

## Prong 2 / LibriPhrase pre-flight verification (2026-08-26/27)

Preparation for scoring the cascade against PhonMatchNet's **published**
LibriPhrase numbers (LP-E 2.80 / 99.29, LP-H 18.82 / 88.52), taken as ground
truth - we never run their model. Because their side is never re-run, a protocol
mismatch could not be absorbed after the fact, so every checkable assumption was
checked first. Ten checks; each passed, was made moot, or produced a finding.

### Documentation that did not survive checking

1. **No LibriPhrase EER harness ever existed in this repo.** The vendored
   `PhonMatchNet_VMS_Project/phonmatchnet/` is an inference-only subset (`model/`,
   `dataset/g2p/`, `google_speech_embedding/`); no `train.py`, `dataset/libriphrase.py`
   or `criterion/`, and absent from every commit in `git log --all` for that path.
   Consequently **LP-E 6.97 / LP-H 28.39 has no backing artifact** - it appears only
   in prose (that project's `README.md:61-63`, `APPROACH_COMPARISON.md`,
   `EXPERIMENTAL_PLAN.md:44` and `:224`) and this ledger had no PhonMatchNet entry
   at all. It was produced on the rented AWS host in June 2026; only the two
   checkpoints came back. `EXPERIMENTAL_PLAN.md` is wrong where it claims a working
   harness, and the B4 cost estimate resting on that claim was optimistic.
   Upstream has no `test.py` either - evaluation lives inside `train.py`.
2. **The 2026-08-19 meeting note's CED row is wrong**, and marked *verified*. It
   lists LP-E 1.7 / 99.84 and LP-H 14.4 / 92.7; those four numbers appear nowhere
   in P1. P1's Table 2 ("(II) Proposed") reports **8.42 / 96.70** and
   **32.90 / 73.58**. P2's own re-implementation of CED lands at 10.48 / 95.63 and
   29.34 / 77.60, corroborating 8-10 / 29-33 rather than 1.7 / 14.4.
   **[Pointer added 2026-09-02, original text above left intact: this entry is
   RETRACTED. 1.7 / 14.4 are the real published numbers of CED (Nishu et al.,
   ICASSP 2024), a different system from P1 (= CMCD). See the 2026-09-02 entry at
   the end of this file.]**
3. **The LibriPhrase repo ships no `data/` directory** - only `libriphrase.py`,
   `utils.py`, `README.md`, `requirements.txt`. An earlier claim that it ships the
   word alignments (so no forced aligner is needed) was false. Moot in the end, see 4.

### The dataset

4. **The complete eval set is public and pre-built**: `charsiu/libriphrase` on
   HuggingFace carries the four `libriphrase_diffspk_all_Nword.csv` files (the exact
   glob PhonMatchNet's loader expects) plus `LibriPhrase_evalset.zip`.
   **136,461 wav files; 136,461 unique CSV-referenced paths; 100.0% resolution in
   both directions.** Every path sits under `train-other-500/`. Unique anchor clips
   divided by 3 gives **4,391 / 2,605 / 467 / 56** for 1/2/3/4 words - P1's published
   episode counts, exactly, for all four lengths, at 27 rows per episode. The data
   also settles the `--maxspk` ambiguity (1166 in the script's defaults vs 1611 in
   its README): **1,166 unique speakers**. This removed the ~30 GB LibriSpeech
   download, the MFA pass, the FLAC conversion, *and* the unseeded-generator
   reproducibility problem - we inherit the same file everyone else uses.
   Durations are 0.50-2.00 s, **median 0.61 s**; 0.4% of rows share a speaker, so
   "diffspk" is near-universal rather than strict.
5. **One CSV row expands into four scored samples** (verified verbatim in upstream
   `dataset/libriphrase.py`): each clip is paired with its **own** transcript
   (label 1) and with the other clip's transcript (label from `target`). All four
   inherit the row's `type` via `anc_pos['type'] = df['type']`, which is why
   filtering `type == diffspk_easyneg` still yields both classes. The
   `diffspk_positive` rows are used only in `both` mode, not for LP-E/LP-H. Scoring
   only the anchor-keyed pairs would leave almost no positives and make EER
   meaningless.
6. **Their EER is batch-averaged, not pooled.** `criterion/utils.py` defines a Keras
   metric that does `score += compute_eer(batch); count += 1; result = score/count`,
   and the eval loop calls it per batch with `GLOBAL_BATCH_SIZE = 2048 *
   num_replicas` and `shuffle=True`. **AUC** uses `tf.keras.metrics.AUC` and *is*
   pooled. So 2.80 / 18.82 are means of ~132 per-batch EERs. `compute_eer` itself is
   `sklearn.metrics.roc_curve` then the mean of fpr and fnr at `argmin|fnr - fpr|`.

### Our side

7. **Whole-clip scoring is required, and is also better.** On 840 short clips
   (keyword TTS variants vs their own rivals, 0.32-1.22 s), the multi-scale sliding
   detector gave **no score at all** to 8 clips - the `pool_windows` -> `n_windows == 0`
   path at `detector.py:132-153`. Whole-clip scored every clip and did better
   overall: pooled **AUC 95.72 / EER 10.04%** vs sliding 95.41 / 11.33%. The
   per-keyword pattern is consistent - the longer the anchor window relative to the
   clip, the more sliding hurts (`administration`, 1.12 s window: 14.1% -> 2.4%;
   `weather`/`party` at 0.51 s: unchanged). Caveat: positives are the keyword's own
   TTS variants, so absolute separation is optimistic and TTS-vs-TTS never crosses
   the synthetic-to-real gap. This was a mechanics test, not a prediction.
8. **The RAV margin alone is the wrong score function for a threshold-free metric.**
   The margin is continuous and well-behaved (90.9% exact-distinct values, median
   consecutive gap 1.7e-4, 0.005% mass at the extremes) but as a *standalone* score
   it degrades a perfect detector: on Sets D/F the detector scores AUC 100.00 / EER
   0.00% while the bare margin gives 97.31 / 2.78%. `outbreaks` shows the mechanism:
   live_2 ("prolonged outbreaks") scores 4.99 but its rival *outcomes* also fires at
   2.49, so the **difference** collapses to -0.06 - below keyword-free chunks whose
   scores are all low and whose difference is therefore small but positive. The
   margin discards the absolute score, which was doing the discriminating. This is
   coherent with RAV's design: it is a gate applied only to windows that already
   cleared the threshold, where a high absolute score is guaranteed. The faithful
   continuous form is
   **`s_cascade = s_det - lambda * max(0, delta - margin)`** - it can only demote,
   is monotone in both inputs, and reduces to the gate at a threshold. It held
   AUC 100.00 / EER 0.00% at lambda in {0.5, 1.0, 2.0}. Lambda is unidentifiable on
   broadcast data (the penalty never binds when the detector is already perfect) and
   must be set on LP-Hard.
9. **Rival arming is healthy; `delta = 0.0` is not a bug.** Across 15 keywords all
   armed 11-16 rivals (median 15), none zero, and the worst-positive-minus-best-rival
   gap was always positive (+0.743 to +7.771, median +2.551). `delta = 0` for 14 of
   15 is `DELTA_FLOOR` clamping a negative computed bar: since `rival_margin_max` is
   negative everywhere, `rival_max + 0.2*(pos_p10 - rival_max)` goes negative.
   `america` is the exception at +0.839, which reproduces its stored value exactly.
   Multi-word anchors were a suspected zero-rival risk; **"she said" armed 16**, so
   the risk did not materialise, though a defensive fallback is still warranted
   because `margins()` raises on an empty rival set.
10. **The whole cost is SpeechT5 synthesis.** Measured marginal cost per keyword
    (models resident): `keyword_generator` ~0-12 s, `cohort_builder` **9.8 s**,
    `rival_builder` **148.1 s** - 94% of the total 157.5 s, and real work
    (16 rivals x 7 voices = **112 TTS renders**, ~1.3 s each on CPU). Cohort
    duration-bucketing saves 6%, not "about half" as first estimated. Scoring is
    negligible: 197.2 ms per unique clip (WavLM forward), 0.99 ms per sample
    (AS-norm), 15.73 ms per sample (`margins()` with 16 rivals) - **1.39 CPU-hours
    for the whole run, 1% of enrollment**. The forward is per *clip*, not per sample
    (16,101 clips carry 47,006 samples), so embeddings cache. LP-Hard needs only
    **1,969** unique keywords against LP-Easy's 7,643, because hard negatives are
    drawn from a narrow phonetic neighbourhood and repeat.

### The finding that changed the code

11. **The pipeline was non-deterministic, and had been for every keyword ever
    built.** Two from-scratch builds of the same keyword agreed on `keyword`,
    `delta`, `n_stream`, `n_tts` and disagreed on **everything derived from audio** -
    `centroid`, `positives`, `window_samples`, cohort `embeddings`, rival `centroids`,
    all four margin stats, and the rival `words` list itself. Root cause: SpeechT5
    applies speech-decoder-prenet dropout at inference **by design** (Tacotron 2
    convention; `SpeechT5SpeechDecoderPrenet._consistent_dropout` calls
    `torch.bernoulli` directly, so `model.eval()` does not disable it and a
    module-level scan shows `training=False`), consuming torch's global RNG - which
    `keyword_generator.py` never seeded, having seeded only NumPy (42/43, plus
    cohort 123 and calibration 777). Isolated confirmation: unseeded renders of the
    same word and x-vector differed by max 3.9e-01; with `torch.manual_seed` they
    were bit-identical. **Fix:** `synthesize()` now seeds torch from a SHA-256 hash
    of `(text, x-vector, TTS_SEED)` immediately before generating - per call rather
    than per process, so clip *N* does not depend on how many clips preceded it and
    changing `--n-augment` or the voice list cannot shift earlier results. Both
    entry points are covered because `rival_builder` imports the same function.
    Re-run verdict: **all 15 arrays identical**, including `window_samples` and
    `words`. The invariant in CLAUDE.md now actually holds. Pre-existing artifacts
    are unaffected because clips are cached, so the **19 keywords on disk (including
    all 16 behind the Set D/E/F results) remain irreproducible until rebuilt** - a
    pending isolated rebuild will establish whether micro-F1 1.000 / 0.982 survives,
    and a change there is a finding, not a repair.

### Decisions taken

- **Scope**: exact PhonMatchNet protocol, **500 anchor classes, seed 777** -
  47,006 samples / 9,030 keywords / 16,101 clips; LP-Hard subset 20,564 / 1,969 /
  7,325. Full evaluation (51,020 keywords) is ~6.6x and deferred, because the
  determinism fix, lambda, and the cohort source all had to be settled first.
- **Compute**: Modal free tier ($30/month, 100 CPU containers, GPU concurrency 10).
  Enrollment on CPU containers; the job is embarrassingly parallel per keyword.
- **C4 cohort constants**: source decided by A/B on **40 anchor classes disjoint
  from the 500** (deciding on the reported data would be test-set selection - the
  very thing P2 does at its section 4.2); cohort size **50** and `top_k` **50**
  unchanged, since cohort > 50 would activate adaptive selection and change the
  system; **per-keyword** cohorts; **stem-family** exclusion for cohort hygiene
  while labels come from their `target` (two independent uses); TTS distractors off.
- `fit_whitener`/`whiten` in `core/scoring.py` are defined but never called, so the
  cohort affects only AS-norm's mean/std.
- Harness: `benchmarks/libriphrase/` (tracked - it produces a paper number);
  outputs to `Journal_Paper/experiments/`; data outside the repo. No second copy of
  the pipeline, and no change to `detector.py` - the scorer calls
  `scoring.asnorm_windows` and `rival_verify.margins` directly.

## Prong 1 pre-flight: the CED / CMCD conflation (2026-09-02)

Desk verification only - no detection run, no new scoring. Preparation for scoring the
cascade against the paper that **created** LibriPhrase. Appended rather than edited into
the 2026-08-26/27 section, because one entry there is being retracted and the history
should show both states.

### Retraction

**Entry 2 of the 2026-08-26/27 section is wrong and is retracted.** It states that the
2026-08-19 meeting note's "CED row" (LP-E 1.7 / 99.84, LP-H 14.4 / 92.7) is wrong because
"those four numbers appear nowhere in P1". The numbers are real. They are the published
results of a **different system that is also called CED**, and the 2026-08-19 note's actual
error was attaching them to P1's arXiv id. Everything else in that entry stands: P1's own
Table 2 does report 8.42 / 96.70 and 32.90 / 73.58, and P2's re-implementation does land at
10.48 / 95.63 and 29.34 / 77.60. Those figures belong to **CMCD**, not CED.

### The two systems, both verified from their own results tables

| | CMCD | CED |
|---|---|---|
| Full name | cross-modal correspondence detector | not expanded in its own paper |
| Authors | Shin, Han, Kim, Chung, Kang (Yonsei / Naver) | Nishu, Cho, Dixon, Naik (Apple) |
| Venue | Interspeech 2022, pp. 1871-1875, DOI 10.21437/Interspeech.2022-580 | ICASSP 2024, pp. 5050-5054, IEEE Xplore 10447547 |
| arXiv | 2206.15400 | 2308.06472 |
| Params | 0.7M | **3.8M** (its own section 3.2; ProKWS and DMA-KWS both misprint 3.6M) |
| LP-E EER / AUC | 8.42 / 96.70 | 1.70 / 99.84 |
| LP-H EER / AUC | 32.90 / 73.58 | 14.40 / 92.70 |
| Introduced LibriPhrase | yes | no |

CMCD's 8.42 / 32.90 is quoted identically by ProKWS, MALEFA and DMA-KWS - three independent
corroborations. CED's LP-H 14.4 / 92.7 is confirmed in its own abstract ("increasing AUC
from 84.21% to 92.7% and reducing EER from 23.36% to 14.4%").

### What the desk pass established

11. **Neither system ever released model code.** CMCD's repository
    (`gusrud1103/LibriPhrase`) is the dataset recipe only; issue #1 asking for the
    implementation has been open and unanswered since 2023-03-08. CED's paper names no
    repository. **Consequence: pooled-vs-batch-averaged EER is unknowable for both.** Only
    P2's is verifiable (batch-averaged, `criterion/utils.py`), which is what our harness
    matched. Our own two forms differ by ~0.04 EER, far below any gap in the table.
12. **The LibriPhrase protocol is publicly contested.** Repository issue #7, opened
    2026-08-05 and unanswered: PhonMatchNet expands **4 trials per CSV row**, DMA-KWS and
    MM-KWS expand **2**, and 40-61% of the resulting trials are duplicate
    (keyword, clip) pairs depending on protocol, with no paper stating whether it
    deduplicated. Affects our already-reported row and P4's prong.
13. **CMCD's episode expansion is reconstructable from the scores already on disk, at zero
    compute.** CMCD describes each episode as 3 positive plus 3 negative pairs. Restricting
    `500_cas_*.csv` to the comparison-clip side (`src` in `com_pos`, `com_neg`) gives
    LP-Easy **8,800 / 8,833** and LP-Hard **4,379 / 4,401** - 49.9/50.1 in both, exactly
    the balance that description predicts. The anchor-keyed side is **not** usable alone:
    it is an identical 2,946 / 8,838 for both difficulties, because an anchor's own
    positives do not depend on which negative it was paired with, so an anchor-only score
    cannot separate LP-Easy from LP-Hard at all.
14. **The per-word-length breakdown (CMCD's Figure 5) is also free.** Counts over the same
    49,981 rows, positives in brackets: LP-Easy 1w 18,827 (7,510), 2w 8,340 (3,336), 3w
    2,190 (876), 4w 60 (24); LP-Hard 1w 13,424 (4,816), 2w 5,626 (1,977), 3w 1,468 (515),
    4w 46 (17). The 4-word bucket is too small for a stable EER and will be reported
    without conclusion; CMCD had the same problem with 56 four-word episodes of 7,519.
15. **CED's confusable-keyword module is close prior art for RAV**, and its ablation moves
    the same way ours does. CED Table 1: confusables cost **0.90** EER on LP-Easy
    (0.80 -> 1.70) and buy **4.00** on LP-Hard (18.40 -> 14.40). Ours: cost **0.35**
    (2.08 -> 2.43), buy **1.94** (25.85 -> 23.91). Two unconnected groups, same public
    data, same direction. This is simultaneously strong external support for RAV's premise
    and a constraint on the novelty claim: theirs is training-time and baked into weights,
    ours is decision-time, synthesised per keyword, with no retraining.

### Decisions taken

- **Rename to CMCD** wherever Shin et al. is meant. Applied to `paper/paper_skeleton.tex`
  (live file only; dated checkpoints left as historical snapshots), `EXPERIMENTAL_PLAN.md`,
  and the meeting notes, which carry correction banners rather than rewrites.
- **Prong order: CMCD first, then CED as a separate follow-on prong.** CED is a valid
  published baseline that beats us on both halves, but it is a second piece of work.
- **Scope of the CMCD prong: zero new compute.** Every planned output is arithmetic over
  `Journal_Paper/experiments/libriphrase/scores/`. Plan and cost:
  `Journal_Paper/meeting_notes_work/2026-09-02-cmcd-comparison.md`.
- **Pre-committed:** the same 500 classes at seed 777, and lambda stays at 4. No
  re-selection of lambda under the new expansions, even if a different value scores better
  on the reported set.
