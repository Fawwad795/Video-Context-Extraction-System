# VMS Keyword Spotting — Approach Comparison

**Audio-Signal-Processing (Siamese) vs. G2P / Query-by-Text (PhonMatchNet)**

This document compares the two solution directions explored for the Video Context
Extraction System's keyword-spotting task: detecting a user-specified keyword in a
live video/audio stream.

---

## 1. Bottom line

| | Audio-SP / Siamese triplet | **G2P / PhonMatchNet** |
|---|---|---|
| Enrollment | synthesize or clip an **audio** anchor | type the keyword as **text** |
| Matching | distance between mean-pooled embeddings | phoneme-sequence cross-attention |
| LibriPhrase-Hard EER | ~44% (triplet-anchor baselines) | **28.4%** |
| VMS chunk `penalty` | **F1 = 0.00** | **F1 = 1.00** |
| VMS chunk `elon` | **F1 = 0.20** | **F1 = 1.00** |
| Verdict | fails on conversational speech | **clear winner** |

**The G2P / query-by-text approach is the recommended direction.** It fixes the core
failure of the Siamese approach and works on the project's own data.

---

## 2. The two approaches

### Audio-Signal-Processing / Siamese triplet (Phase 1–2)
- A frozen `wav2vec2` backbone + a projection head trained with **Triplet Margin Loss**
  on MLCommons human speech.
- At inference: build an **audio anchor** for the keyword (TTS-synthesized or clipped
  from a human), slide a window over the stream, and compare embeddings by L2/cosine
  distance against a threshold.
- Enhancements tried: multi-voice **prototype anchor**, **adaptive S-norm** scoring,
  per-keyword threshold **calibration**, audio **denoising**, and a domain-adversarial
  (**GRL**) retrain.

### G2P / Query-by-Text (PhonMatchNet, Phase 3)
- Upstream model [ncsoft/PhonMatchNet](https://github.com/ncsoft/PhonMatchNet)
  (Interspeech 2023). The keyword is converted to **phonemes** (G2P) and embedded; the
  audio is encoded (raw + Google Speech Embedding); a **cross-attention matcher** aligns
  the phoneme sequence to the audio and outputs a detection probability.
- No audio anchor is ever created — the keyword is enrolled purely as text.

---

## 3. Results

### Standardized benchmark — LibriPhrase (EER, lower is better)
| Split | Triplet audio-anchor baseline | **PhonMatchNet (ours)** | PhonMatchNet (paper, +360h) |
|---|---|---|---|
| LP-Easy | ~32% | **6.97%** | 2.80% |
| LP-Hard | ~44% | **28.39%** | 18.82% |

### The project's own data — VMS stream chunks (F1, higher is better)
Conversational debate chunks, ground-truth from Whisper word timestamps:

| Keyword | Siamese triplet | **PhonMatchNet** |
|---|---|---|
| penalty (in live_4, live_7) | **0.00** | **1.00** |
| elon (in live_0) | **0.20** | **1.00** |

On scripted news speech the Siamese approach did better (`absolutely` F1 0.57,
`heat` F1 0.80) — but it collapsed on faster conversational speech, which is the harder
and more realistic case.

---

## 4. Why the Siamese approach failed — the ranking inversion

The keyword detector only works if the true keyword scores **higher** than every other
word. On conversational speech, the Siamese model **ranked confusable words above the
true keyword**:

- For `penalty`, the segment *"appropriate"* scored a higher cosine to the anchor (0.984)
  than the actual *"penalty"* (0.949).
- For `elon`, five unrelated segments out-ranked the true *"Elon"*.

No threshold can fix an inverted ranking. We confirmed this is structural:
- **Threshold tuning** (AS-norm + calibration) — couldn't fix it.
- **Denoising** the audio — improved the true keyword's rank (e.g. `penalty` #5 → #2) but
  still left impostors on top.
- **Retraining** the projection head with TTS-mixed triplets + a gradient-reversal domain
  classifier — gained only ~1 point of AUC (architecture ceiling).

Root causes: (a) the **synthetic-to-real domain gap** (a TTS anchor sits far from human
speech), and (b) a single **mean-pooled** whole-word vector is phonetically blurry, so
similar-sounding words collapse together.

## 5. Why G2P / PhonMatchNet wins

- **No audio anchor** → the synthetic-to-real gap disappears (text is domain-free).
- **Phoneme-level matching** → it checks whether the ordered phoneme pattern
  `/p-eh-n-ah-l-t-iy/` appears in the audio; *"appropriate"* simply doesn't contain it,
  so it's rejected regardless of overall acoustic similarity.
- Result: the true keyword **ranks #1** on the VMS chunks (penalty 0.994 > all impostors;
  elon 0.515 > all impostors), giving F1 = 1.00 where the triplet method got 0.00 / 0.20.

---

## 6. Honest limitations

- **Small VMS test:** 2 keywords, 7 unique chunks — a strong signal, not yet a
  statistically robust number. Needs evaluation on more keywords and streams.
- **Per-keyword thresholds differ** (penalty fires ~0.99, elon ~0.51), so a single global
  threshold isn't perfect — production needs per-keyword calibration (the Phase-1 AS-norm
  / calibration machinery transfers directly).
- **LP-Hard 28.4%** trails the paper's 18.8% because we trained on `train-clean-100` only;
  adding `train-clean-360` is the lever to close that gap.

## 7. Recommendation

1. **Adopt the G2P / query-by-text approach** as the project's keyword spotter.
2. Strengthen the evidence: evaluate on more keywords/streams; add per-keyword threshold
   calibration; optionally add `train-clean-360` to chase the paper's numbers.
3. Integrate the trained model into the live VMS pipeline.

Artifacts: [`Siamese_VMS_Project/`](Siamese_VMS_Project/) (audio-SP approach),
[`G2P_VMS_Project/`](G2P_VMS_Project/) (PhonMatchNet approach + trained model).
