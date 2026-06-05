# Results — Siamese Keyword-Spotting Extension

Held-out evaluation on the **official Google Speech Commands v2 test split** (35 words,
unseen speakers). Every model is scored on the *same* same-word / different-word test pairs,
so the comparison is apples-to-apples. The task: *given two short audio clips, decide whether
they are the same spoken word* — across different speakers.

## Headline: P0 (correlation) vs P2 (Siamese)

| Method | Params | Checkpoint | **ROC-AUC** | Accuracy | F1 | Threshold |
|---|---|---|---|---|---|---|
| **P0 — cross-correlation** (`baseline_correlation.py`) | — | — | **0.555** | 0.540 | — | 11.6% match |
| **P2 — Siamese, reduced/edge** (`model.py` `reduced`) | **0.14M** | **0.58 MB** | **0.966** | 0.908 | 0.908 | cos 0.806 |
| **P2 — Siamese, full** (`model.py` `full`) | 1.50M | 6.0 MB | **0.987** | 0.958 | 0.957 | cos 0.730 |

**Lift over the correlation baseline:** +0.41 AUC (edge) / **+0.43 AUC (full)** — from
near-chance to near-perfect.

## What this shows

- **Cross-correlation is near-chance at cross-speaker word identity.** Same-word vs
  different-word matching scores were 37.8% vs 37.6% (a 0.2-point gap); AUC 0.555 ≈ a coin
  flip. Raw waveforms of the same word spoken by different people simply don't line up — the
  exact brittleness the project report flagged.
- **The learned Siamese embedding fixes this.** Mapping audio to a 64–128-d embedding and
  comparing by cosine similarity reaches AUC 0.987 (full) / 0.966 (edge): the network learned
  *word identity*, invariant to speaker/pitch/timing.
- **The edge model nearly matches the full model.** 10× fewer parameters and a ~10× smaller
  checkpoint (0.58 MB) cost only ~0.02 AUC / ~0.05 accuracy — confirming a lightweight model
  is deployable on the Jetson without giving up much, the central premise of the ideation.

## Training setup

- Data: 50,000 contrastive pairs/epoch (1 positive : 2 negatives), official train split.
- Loss: contrastive (Hadsell–Chopra–LeCun) on Euclidean distance of L2-normalized embeddings.
- Optimizer: Adam, lr 1e-3, 15 epochs, batch 256.
- Hardware: NVIDIA L4 (AWS), ~40 s/epoch. Decision threshold calibrated on the validation
  split, reported on the test split.
- Reproduce: `python -m siamese.train --model {full|reduced} --epochs 15 --workers 4`
  then `python -m siamese.eval_checkpoint --ckpt siamese/checkpoints/siamese_<model>_v1.pt`.

## Caveats / honest notes

- The report's **72%** was a different setup on TIMIT (a synthesized keyword matched *inside*
  a longer clip); it is **not** directly comparable to this cross-speaker pair task. Our P0 is
  the fair, internally consistent bar for what the Siamese model does.
- Speech Commands clips are clean, isolated single words; broadcast audio is continuous and
  noisier. P3 will measure robustness vs SNR/accent and the synthetic↔real reference gap before
  any deployment claims.

## P3 — robustness & deployment realism (full model)

Deployment-relevant stress tests on the held-out test set (`evaluate_p3.py`, `p3_prototype.py`).

**A. Noise robustness** — background noise added to *both* clips of each pair (conservative),
AUC is threshold-free:

| SNR | clean | 20 dB | 15 dB | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|---|---|
| AUC | 0.988 | 0.984 | 0.978 | 0.971 | 0.952 | **0.900** |
| acc @ clean threshold | 0.962 | 0.945 | 0.932 | 0.915 | 0.883 | 0.812 |

Graceful degradation: even at **0 dB** (noise as loud as the speech) AUC is 0.900 — still far
above the correlation baseline's *clean* 0.555. (`artifacts/p3_robustness_full.png`)

**B. Retrieval mAP:** **0.835** (1,400 clips, 40/word) — same-word clips rank at the top.

**C. Synthesized-keyword (prototype) vs real reference** — match 1,050 real test clips against
35 keyword references; 35-way top-1 (chance = 0.029):

| Reference | Top-1 acc | AUC |
|---|---|---|
| **Synthetic prototype** (7 accents averaged) | **0.877** | 0.993 |
| Real reference (real train clips) | 0.886 | 0.993 |
| synthetic → real gap | **0.009** | 0.000 |

**Key finding:** the synthesized keyword prototype matches real speech *as well as a real
recording* (~1-point gap, identical AUC). The synthetic↔real domain gap — the top risk in the
ideation — is negligible, so the VMS needs **no real enrollment audio**: keep the existing TTS
keyword, average its 7 accents into one prototype, and match by cosine similarity.

## P4 — VMS integration + live YouTube test

The detector was wired into the VMS as a **parallel path** (`Research/siamese_vms_detect.py`,
`Stream*_siamese_detect.py`, `VMS_DETECTOR=siamese` switch) and run end-to-end on a live
stream: synthesize keyword → download 5-s chunks → embed → cosine vs prototype → save context
clip + timestamp log. **The plumbing works** — the pipeline ran live, built the "journalism"
prototype, scored chunks, and produced the same outputs as the original detector.

**The live test exposed an over-detection problem on continuous broadcast speech** that the
isolated-word evaluation (P2/P3) could not reveal. Against the "journalism" prototype, the
max-over-window cosine across 29 live chunks was: **min 0.752, median 0.888, max 0.985** — so
ordinary broadcast speech sits near the cutoff, and 26/29 chunks fired at threshold 0.80
(far too many for a single keyword).

**Why (the isolated-word → continuous-speech gap):**
1. Trained/evaluated on *isolated single words*; broadcast audio is *continuous*, so every 1-s
   window holds word fragments that embed near *some* word.
2. Sliding-window **max** over ~9–10 windows/chunk biases the score upward.
3. The model has no "background / no-keyword" class — it only learned word-vs-word similarity,
   so it cannot *reject* non-keyword speech.

**Path forward (P4.5 — make it deployment-grade on broadcast audio):**
- Calibrate the threshold on broadcast-like audio with continuous-speech **hard negatives**
  (not isolated-word pairs).
- Fine-tune with negatives drawn from continuous speech / other words (the biggest lever).
- Better segmentation: VAD or word-boundary windows instead of arbitrary 1-s slices; and/or an
  open-set "background prototype" rejection (require keyword score ≫ background score).

Net: the integration is mechanically complete and reusable; reaching deployment quality on
real broadcast audio needs continuous-speech calibration/training — done in P4.5.

## P4.5 — dense-speech hard-negative fine-tune

Refined cause: the P2 model maps arbitrary DENSE-speech windows near word prototypes, so with
the sliding-window max over ~9 windows/chunk a ~7% per-window false rate compounds into
broadcast over-detection. Concatenated Speech Commands negatives were too easy (the P2 model
already scored FPR ~0.007 on them), so we fine-tuned (warm-start from P2) with random
**LibriSpeech** windows as dense hard negatives, plus keyword-in-context positives to hold recall.

**Controlled held-out test (LibriSpeech negatives, thr 0.80):**

| Metric | Before (P2) | After (P4.5) |
|---|---|---|
| Continuous **FPR** | 0.070 | **0.007** (10× lower) |
| Continuous **TPR** | 0.960 | **0.979** |
| Isolated-word AUC | 0.987 | 0.985 (retained) |

10× fewer per-window false positives, recall slightly *up*, isolated accuracy held. Checkpoint
`siamese_full_p45.pt` is now the VMS detector default.

**On the 29 real broadcast chunks (thr 0.80):** over-detection roughly halved — "journalism"
26/29 → **10/29**; unrelated words also fell (right 5→2, stop 6→4, house 3→2). Scores compressed
(journalism median 0.888 → 0.749), so the p45 model wants a **recalibrated threshold (~0.72–0.75)**
for broadcast.

**Live re-test + manual check — CRITICAL correction.** Re-running the live stream with the p45
model at threshold 0.73 dropped firing from 26/29 (P2 @ 0.80) to 11/29. **But listening to the 11
detected chunks confirmed NONE actually contain "journalism" — they are all false positives.**
So P4.5 lowers the false-alarm *count* but does not solve detection for this keyword.

**The real limitation (uncovered by testing a real keyword): "journalism" is out-of-distribution.**
The model is trained only on the 35 short, common Speech Commands words, and *every* headline
number above (AUC 0.987, prototype top-1 0.877, the 10× FPR drop) was measured on those SAME 35
words — the evaluation's blind spot. For an arbitrary real keyword like "journalism":
- SpeechT5 mis-synthesizes it (clips ~0.8 s, garbled; cosine to synth-"journal" only 0.55), so the
  prototype is poor; and
- the model has never seen long/multi-syllable words, so the embedding is unreliable and lands
  near generic broadcast speech → false positives.

**What these results do and don't show:**
- DO: Siamese ≫ correlation, noise robustness, and synthetic-prototype matching — **for short,
  in-vocabulary keywords** (the 35 SC words). The VMS integration works mechanically.
- DON'T: generalization to **arbitrary real-world keywords**. "journalism" fails end-to-end.

**To actually support real keywords (future work):**
- train on a BROAD vocabulary (thousands of words, or subword/phonetic units), not 35 words;
- faithful TTS for the full keyword (fix SpeechT5 early-stopping, or use a stronger TTS);
- variable-length handling for multi-syllable words (the fixed 1 s window is too short);
- evaluate on OUT-OF-vocabulary keywords + real labeled broadcast audio — never the training words.
