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

## Next (P3)

Robustness curves (accuracy vs SNR, vs accent), retrieval **mAP**, and the synthesized-keyword
(prototype) vs real-reference comparison — the deployment-relevant evaluation.
