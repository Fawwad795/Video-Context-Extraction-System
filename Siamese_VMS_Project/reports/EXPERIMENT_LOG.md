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

| Date | Keyword | Set | Anchor | Model | Threshold | TP/FP/FN | P / R / F1 | Artifact |
|---|---|---|---|---|---|---|---|---|
| 2026-06-10 | absolutely | A | TTS centroid | v1 baseline | 1.741 | — | F1 = 0.57 | `logs/archive/detections_absolutely_2026-06-10_tts_baseline_oldchunks.json` |
| 2026-06-10 | heat | A | TTS centroid | v1 baseline | 2.244 | — | F1 = 0.80 | `logs/archive/detections_heat_2026-06-10_tts_baseline_oldchunks.json` |
| 2026-06-11 | penalty | B | TTS centroid | v1 baseline | 2.464 | 0/4/2 | F1 = 0.00 | `logs/archive/detections_penalty_2026-06-11_tts_baseline_oldchunks.json` |
| 2026-06-15 | elon | B | TTS centroid | v1 baseline | 1.252 (run 1) / 1.510 (run 2) | — | ~0.20 | `logs/archive/detections_elon_2026-06-15_run{1,2}_*.json` (two same-day runs, recalibrated threshold; kept both) |
| 2026-07-02 | penalty | B | TTS centroid (baseline rerun) | v1 baseline | 1.594 | 0/7/2 | **F1 = 0.00** — true chunks scored 0.96, below all 7 FPs; "appropriate" fired at cos 0.985 | overwritten by the kNN-VC rerun; per-window hits preserved in `logs/archive/timestamps_penalty_2026-06-11_and_2026-07-02_runs.txt` |
| 2026-07-02 | penalty | B | **kNN-VC converted** | v1 baseline | 2.166 | 2/4/0 | **P 0.33 / R 1.00 / F1 = 0.50** | `logs/archive/detections_penalty_2026-07-02_knnvc_anchor_oldchunks.json` |
| 2026-07-02 | administration | C | **kNN-VC converted** | v1 baseline | 2.341 | 1/3/0 | **P 0.25 / R 1.00 / F1 = 0.40** | `logs/detections_administration_unverified.json` |
| 2026-07-02 | administration | C | kNN-VC + **phoneme verification** | v1 baseline + wav2vec2-espeak CTC | 2.341 / tau 0.716 | 1/0/0 | **P 1.00 / R 1.00 / F1 = 1.00** | `logs/archive/detections_administration_2026-07-02_verified_oldchunks.json` |
| 2026-07-02 | washington | C | **kNN-VC converted** (detector alone) | v1 baseline | 2.610 | 1/2/0 | P 0.33 / R 1.00 / F1 = 0.50 — and the live_9 "TP" was accidental: windows fired on "understanding" at 2.0s, not "washington" at 4.2s | `logs/detections_washington_unverified.json` |
| 2026-07-02 | washington | C | kNN-VC + **phoneme verification + candidate rescue** | v1 baseline + wav2vec2-espeak CTC | 2.610 / tau 0.604 | 1/0/0 | **P 1.00 / R 1.00 / F1 = 1.00** — true window at 4.2s rescued from below the embedding threshold with phone-sim 1.00; "understanding" (0.38) and all confusables (<=0.50) rejected | `logs/archive/detections_washington_2026-07-02_verified_oldchunks.json` |
| 2026-07-02 | **russia** | **D** | kNN-VC + phoneme verification | full cascade | 1.408 / tau 0.718 | 1/0/0 | **P 1.00 / R 1.00 / F1 = 1.00** | `logs/detections_russia.json` (current) |
| 2026-07-02 | **weather** | **D** | kNN-VC + phoneme verification | full cascade | 2.126 / tau 0.750 | 1/0/0 | **P 1.00 / R 1.00 / F1 = 1.00** | `logs/detections_weather.json` (current) |
| 2026-07-02 | **scotland** | **D** | kNN-VC + phoneme verification | full cascade | 1.793 / tau 0.707 | 3/1/0 | P 0.75 / R 1.00 / F1 = 0.86 — FP on live_6 ("south of Ireland") crossed tau by 0.003 (phone-sim 0.71 vs 0.707) | `logs/detections_scotland.json` (current) |
| 2026-07-02 | **ireland** | **D** | kNN-VC + phoneme verification | full cascade | 1.581 / tau 0.839 | 2/0/1 | P 1.00 / R 0.67 / F1 = 0.80 — FN on live_6 (candidate windows missed the spoken instant, best phone-sim only 0.40); correctly rejected the ASR-confused "Island" chunk (live_5) | `logs/detections_ireland.json` (current) |
| **2026-07-02** | **4-keyword micro-avg** | **D** | kNN-VC + phoneme verification | full cascade | — | 7/1/1 (40 chunk-decisions) | **P 0.875 / R 0.875 / F1 = 0.875** (macro-avg F1 = 0.915) | first full validation run on a duplicate-free chunk set - see "Set D validation" below |
Note: `detector.py` labels every `.npz` anchor "TTS prototype centroid" in the
JSON; for the 2026-07-02 rows the anchor was actually the kNN-VC-converted
centroid (baseline TTS anchors backed up as `keywords/<kw>_anchor_tts.npz`).

## Set D validation (the pipeline-correctness claim for the paper)

Purpose: confirm the full three-stage cascade (kNN-VC anchor conversion ->
Siamese AS-norm detector -> phoneme-CTC verification) generalizes to content
it has never seen, on the first chunk set collected with the duplicate-free
downloader (see [[vms-chunk-dedup-fix]]). Set D is a Sky News weather
bulletin segueing into a news bulletin - a different register from every
prior chunk set (scripted news / conversational debate / correspondent
report), so this is a genuine out-of-distribution check, not a rerun on
familiar material.

Four keywords were chosen deliberately to span difficulty, not cherry-picked
for a good score:

- **russia**, **weather** - rare (a single true chunk each), phonetically
  distinct - the "does it work at all on a fresh domain" control.
- **scotland** - frequent (3 true chunks) - recall stress test under
  repetition.
- **ireland** - frequent (3 true chunks) **and** whisper-tiny itself
  mis-transcribes one instance as "Island" (live_5) - a near-homophone
  stress test for the phoneme verifier, since /aɪɚlənd/ vs /aɪlənd/ differ
  by a single phone.

| Keyword | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| russia | 1 | 0 | 0 | 9 | 1.00 | 1.00 | 1.00 |
| weather | 1 | 0 | 0 | 9 | 1.00 | 1.00 | 1.00 |
| scotland | 3 | 1 | 0 | 6 | 0.75 | 1.00 | 0.86 |
| ireland | 2 | 0 | 1 | 7 | 1.00 | 0.67 | 0.80 |
| **Micro-avg (40 decisions)** | 7 | 1 | 1 | 31 | **0.875** | **0.875** | **0.875** |
| **Macro-avg** | | | | | **0.938** | **0.918** | **0.915** |

Both errors are explainable, not systemic:
- scotland's FP (live_6, "...south of **Ireland** can expect...") passed the
  phoneme gate by a 0.003 margin (phone-sim 0.71 vs tau 0.707) - a genuine
  near-miss the verifier almost caught, not a domain-gap failure.
- ireland's FN is the same chunk (live_6): the detector's candidate windows
  didn't land squarely on the spoken instant, so the best phonetic match was
  only 0.40. Precision stayed perfect elsewhere, including a correct reject
  of the ASR-confused "Island" chunk (live_5) - the verifier is not just
  matching orthography, it is matching the actual phone sequence spoken.

**Claim this supports:** the three-stage pipeline achieves F1 ~0.88-0.92
(micro/macro) zero-shot keyword spotting on live-stream audio outside its
enrollment domain, across both rare and frequent keywords, with remaining
errors traceable to specific near-threshold cases rather than the systemic
F1=0.00 domain-gap failures documented earlier in this log (rows above,
2026-06-11 / 2026-07-02 TTS-baseline penalty).

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
3. **Phoneme verification closes the precision gap.** A second-stage check
   (`pipeline/verify_detections.py`) CTC-decodes each candidate window with
   facebook/wav2vec2-lv-60-espeak-cv-ft and requires the anchor's phone
   sequence (infix edit distance vs. references decoded from the anchor
   clips; tau calibrated at the 99th percentile of random stream windows).
   On 'administration' it kept the true detection (phone-sim 0.92) and
   dropped all false positives (0.31 each): F1 0.40 -> 1.00. The two views
   fail independently - embeddings capture holistic acoustic shape, the CTC
   decoder capture the phone sequence - so their conjunction filters
   confusables that pass either test alone.
4. **Candidate rescue + reference-informed tau make the cascade robust.**
   The detector saves its top-8 NMS windows per chunk regardless of
   threshold; the verifier checks them too, so a true keyword window whose
   embedding score fell below the threshold can be rescued (washington in
   live_9: detector fired on "understanding" instead, the verifier rejected
   it at 0.38 and rescued the real occurrence at 4.2s with phone-sim 1.00).
   tau = midpoint(negative percentile, reference leave-one-out
   self-similarity), floored at 0.5 - near-miss confusables sharing half the
   keyword's phones (0.50-0.62) fall below it in both keywords tested.
5. Anchor references decode cleanly in both domains
   (/æ d m ɪ n ɪ s t ɹ eɪ ʃ ə n/ from TTS and from kNN-VC-converted
   clips alike), confirming the phoneme space is domain-invariant where the
   embedding space was not.

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
