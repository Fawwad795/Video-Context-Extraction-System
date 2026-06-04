# Siamese Keyword-Spotting Extension

A learned audio-similarity matcher that replaces the hand-crafted cross-correlation
detector in the original VMS (`../Research/`). Full design + rationale:
[`../Extension/Siamese_Network_Extension_Ideation.md`](../Extension/Siamese_Network_Extension_Ideation.md)
*(kept local / gitignored)*.

> **Idea in one line:** embed the live audio chunk and the (multi-accent, synthesized)
> keyword with a shared CNN, and score them by **cosine similarity** in the learned space —
> replacing `np.correlate` + a hand-tuned 70% threshold.

## Roadmap status
| Phase | What | Status |
|---|---|---|
| P1 | Data pipeline: Speech Commands loader, pair generator, augmentation, TTS bridge | ✅ done |
| P0 | Correlation baseline on the held-out test split | ✅ done — AUC **0.555** |
| P2 | Siamese CNN (full + reduced) + contrastive loss, trained on the AWS L4 GPU | ✅ done — full AUC **0.987** |
| P3 | Offline benchmark (mAP, robustness vs SNR/accent, synth↔real reference) | next |
| P4 | Drop-in integration into the VMS detector | pending |
| P5 | Edge export (ONNX/TensorRT) + on-device latency/RAM | pending |

See [RESULTS.md](RESULTS.md) for the full P0 vs P2 comparison.

## Layout
```
siamese/
├── config.py        # audio + pairing hyperparameters (one source of truth)
├── audio.py         # waveform -> fixed-size log-mel spectrogram
├── datasets.py      # Speech Commands wrapper + same/different PairDataset
├── augment.py       # background noise, SpecAugment, pitch/time-stretch
├── tts_bridge.py    # SpeechT5+HiFiGAN synthesis -> synthetic<->real positive pairs
├── verify_p1.py     # P1 sanity check (stats + spectrogram figure)
├── requirements.txt
├── data/            # (gitignored) Speech Commands v2 + tts_cache
├── artifacts/       # (gitignored) figures
└── checkpoints/     # (gitignored) trained models
```

## Setup
```bash
pip install -r siamese/requirements.txt
# Download Speech Commands v2 (~2.4 GB) into siamese/data/:
python -c "import torchaudio; torchaudio.datasets.SPEECHCOMMANDS(root='siamese/data', download=True)"
```

## Run the P1 check (from repo root)
```bash
python -m siamese.verify_p1
```
Prints split sizes + pair balance + batch shapes and writes
`siamese/artifacts/p1_pipeline_check.png`.

## (Optional) build the synthesized keyword cache
Heavy — downloads SpeechT5; best run on the GPU box:
```bash
python -m siamese.tts_bridge --all-speech-commands
```

## Design notes
- **Front-end:** 64-bin log-mel, 25 ms / 10 ms windows (FBank-style; Li & Song 2021 found
  FBank/log-mel beats MFCC for DNN matching).
- **Splits:** the dataset's *official* train/val/test lists, so the held-out test set is
  reproducible and the P0/P3 comparison is fair.
- **Generalization:** the 35 SC words are only training vocabulary; a metric/Siamese model
  learns a similarity function and transfers to arbitrary unseen keywords at inference.
