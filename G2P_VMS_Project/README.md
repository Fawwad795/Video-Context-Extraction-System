# G2P / Query-by-Text VMS Keyword Spotting (PhonMatchNet)

This is the **G2P (grapheme-to-phoneme) open-vocabulary** keyword-spotting approach
for the Video Context Extraction System. It replaces the audio-anchor design of the
[Siamese triplet approach](../Siamese_VMS_Project/) with a **text-enrolled** model:
the user types a keyword, it is converted to phonemes, and the model aligns that
phoneme sequence against the live audio — no synthesized/recorded audio anchor needed.

## Why this approach

The Siamese triplet method (Phase 1/2) failed on conversational speech because it
compared a single mean-pooled audio-anchor embedding by distance, and **confusable
words out-ranked the true keyword** (e.g. on the VMS chunks: `penalty` F1 = 0.00,
`elon` F1 = 0.20). Enrolling the keyword as **text** and matching at the **phoneme
level** fixes that ranking failure.

## Architecture (PhonMatchNet)

Upstream model: [ncsoft/PhonMatchNet](https://github.com/ncsoft/PhonMatchNet)
(Interspeech 2023), PyTorch branch.

```
keyword text ── G2P ──► phoneme embedding ─┐
                                           ├─► cross-attention matcher ─► P(keyword)
live audio ──► Google Speech Embedding ────┘            (+ raw audio → log-mel)
```

- **Text branch:** grapheme-to-phoneme (g2p_en) → phoneme embedding sequence.
- **Audio branch:** raw waveform (→ internal log-mel) **and** a precomputed
  Google Speech Embedding (96-d) — the model's `audio_input='both'` two-stream encoder.
- **Matcher:** stacked self/cross-attention extractor + GRU discriminator → detection
  probability.

## Training (done on AWS)

Trained on **LibriPhrase** (generated from LibriSpeech `train-clean-100`) on an AWS
`g6.xlarge` (NVIDIA L4). Eval used the official LibriPhrase test set.

| Split | EER (best) | AUC |
|-------|-----------|-----|
| LP-Easy | 6.97% | 0.980 |
| LP-Hard | 28.39% | 0.782 |
| Overall | 19.15% | 0.884 |

(LP-Hard ~28% vs ~44% for triplet audio-anchor baselines; the gap to the paper's
18.8% is from training on `train-clean-100` only, not `100h + 360h`.)

> Note: the upstream `train.py` saves checkpoints with `safetensors`, which silently
> drops the GRU recurrent weights (shared-buffer bug). The trained model here was saved
> with `torch.save(state_dict)` instead — it loads with all weights intact.

## Result on the VMS chunks (apples-to-apples vs Siamese)

| Keyword | Siamese triplet | **PhonMatchNet (this)** |
|---------|-----------------|-------------------------|
| penalty | F1 = 0.00 | **F1 = 1.00** |
| elon | F1 = 0.20 | **F1 = 1.00** |

The true keyword now ranks **#1** in both cases (penalty: true 0.994 > all impostors;
elon: true 0.515 > all impostors). Caveats: small test (2 keywords, 7 unique chunks),
and per-keyword thresholds differ — production use needs per-keyword calibration.

## Files

- `vms_gemb.py` — **Stage A** (run in the TF `preprocess` docker): resample the VMS
  chunks to 16 kHz, slide 1.5 s windows, compute Google Speech embeddings → `vms_gemb.npz`.
- `vms_infer.py` — **Stage B** (run in the `udkws_torch` docker): load the trained model,
  G2P-embed the keyword, score each window, max-pool per chunk, sweep threshold for F1.
- `phonmatchnet_model/phonmatchnet_epoch13.pt` — the trained model (best overall EER,
  loads with `BaseUKWS(vocab=72, audio_input='both', text_input='g2p_embed',
  stack_extractor=True, ...)`).

## Reproduce

Both scripts run inside the upstream PhonMatchNet docker images on the training host
(repo mounted at `/home`, chunks at `/c`, output at `/out`):

```bash
# Stage A — Google embeddings (TF docker)
docker run --rm -v <repo>:/home -v <chunks>:/c -v <out>:/out preprocess \
    bash -c 'cd /home/google_speech_embedding && cp /home/vms_gemb.py . && python vms_gemb.py'

# Stage B — model inference + F1 (torch docker)
docker run --rm -v <repo>:/home -v <out>:/out udkws_torch \
    bash -c 'cd /home && python nltk_setup.py && python vms_infer.py'
```
