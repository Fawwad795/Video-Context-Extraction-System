# Siamese KWS — Consolidated Experiment Log

One place for every detection experiment run with the Siamese pipeline, with
pointers to the surviving artifacts. Complements `SIAMESE_PROGRESS_REPORT.md`
(narrative + architecture); this file is the results ledger for the write-up.

## Chunk sets

| Set | Period in `audios/` | Content | Status |
|---|---|---|---|
| A — scripted news | ≤ 2026-06-10 | BBC-style scripted delivery | retired, chunks deleted |
| B — conversational debate | 2026-06-10 → 2026-07-02 | Elon Musk debate panel ("penalty", "elon", "appropriate") | retired, chunks deleted |
| C — Iran deal report | since 2026-07-02 | correspondent report ("administration", "immigration") | **current** (`audios/` + `audios/transcripts.txt`) |

All experiments are 10 chunks, chunk-level ground truth from whisper-tiny
(`validate_detection.py`). Sets B and C contain duplicated chunks (e.g. B:
live_3≡6, live_4≡7, live_5≡8), so counts are directional, not statistical.

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
| 2026-07-02 | administration | C | kNN-VC + **phoneme verification** | v1 baseline + wav2vec2-espeak CTC | 2.341 / tau 0.716 | 1/0/0 | **P 1.00 / R 1.00 / F1 = 1.00** | `logs/detections_administration.json` (current) |
| 2026-07-02 | washington | C | **kNN-VC converted** (detector alone) | v1 baseline | 2.610 | 1/2/0 | P 0.33 / R 1.00 / F1 = 0.50 — and the live_9 "TP" was accidental: windows fired on "understanding" at 2.0s, not "washington" at 4.2s | `logs/detections_washington_unverified.json` |
| 2026-07-02 | washington | C | kNN-VC + **phoneme verification + candidate rescue** | v1 baseline + wav2vec2-espeak CTC | 2.610 / tau 0.604 | 1/0/0 | **P 1.00 / R 1.00 / F1 = 1.00** — true window at 4.2s rescued from below the embedding threshold with phone-sim 1.00; "understanding" (0.38) and all confusables (<=0.50) rejected | `logs/detections_washington.json` (current) |

Note: `detector.py` labels every `.npz` anchor "TTS prototype centroid" in the
JSON; for the 2026-07-02 rows the anchor was actually the kNN-VC-converted
centroid (baseline TTS anchors backed up as `keywords/<kw>_anchor_tts.npz`).

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

- `logs/detections_<kw>.json` — latest run on the **current** chunk set only.
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
