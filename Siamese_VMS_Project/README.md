# Siamese Video Context Extraction System

This project is a state-of-the-art **Zero-Shot Audio Context Extraction Pipeline**. It is designed to download YouTube videos, extract their audio into chunks, and use massive-vocabulary Metric Learning to search for specific spoken keywords without relying on Speech-to-Text transcription during live inference.

By treating audio keyword spotting as an **Image Recognition Similarity Problem** (comparing Phonetic Embeddings in abstract tensor space), the system can instantly identify target words regardless of who is speaking them, while perfectly rejecting background noise.

## Architecture

The system relies on a **Siamese Neural Network** with the following architecture:
- **Feature Extractor (Backbone):** `facebook/wav2vec2-base` (Pre-trained CNN layers to abstract raw audio arrays into robust acoustic features).
- **Metric Layer (Projection Head):** A custom linear projection head that compresses the 768-dimensional Wav2Vec2 outputs down to a highly optimized 256-dimensional embedding space.
- **Loss Function:** `TripletMarginLoss(margin=1.0)`. During training, the network was fed an anchor (Word A), a positive sample (Word A spoken by someone else), and a negative sample (Word B). It was penalized until the positive samples clustered together tightly and the negative samples were pushed far away (distance > 1.0).

## Training Details & Success

The projection head was trained entirely on the massive **MLCommons/ml_spoken_words** dataset.
- **Dataset Size:** 5.1 Million short human audio clips.
- **Hardware:** AWS `g6.xlarge` (NVIDIA L4 Tensor Core GPU) with a 500GB EBS volume attached for massive caching.
- **Convergence:** The network trained for 50 Epochs and successfully converged to a validation loss of `~0.29`, proving it successfully learned how to abstract the phonetic fingerprints of human speech.

## The Detection Pipeline

The `detector.py` application executes the following sequence during live tracking:
1. **The Sliding Window:** It creates a window matching the exact millisecond duration of the target keyword. 
2. **Ultra-Fine Granularity:** It slides that window across the live video audio chunks in `50ms (0.05s)` increments. This guarantees perfect temporal alignment over the exact fraction of a second where the human is speaking the target word.
3. **The L2 Distance Check:** For every 50ms slice, it generates an embedding and compares it to the anchor keyword's embedding. If the Euclidean (L2) distance drops below the `1.25` threshold, it triggers an "AI Match Found!" alert with the exact timestamp.

## Critical Findings: The Acoustic Domain Gap

During development, we tested two different anchor strategies:
1. **Synthetic TTS Anchor (`absolutely.wav`):** A computer-generated robot voice saying the target word.
2. **Real Human Anchor (`human_absolutely_real.m4a`):** A biological human saying the target word.

**The Result:** The system flawlessly found the word using the Human Anchor, but found **Zero Matches** using the Synthetic Anchor. 

**The Reason:** Because the network trained on 5.1 Million *human* examples, it learned an incredibly tight, optimized acoustic box for biological human vocal tracts. It learned to expect organic breath noise, biological jitter, and dynamic formant shifting. When we fed it a synthetic, mathematically crisp TTS voice, the CNN immediately realized it belonged to a completely different acoustic domain, resulting in massive L2 Distance penalties (`1.60+`). 

**Conclusion:** For optimal Zero-Shot detection, the Anchor reference must belong to the same acoustic domain as the target audio (Human to Human).

## Phase 1 Upgrade: Prototype Anchor + Adaptive S-Norm (2026-06)

Two production fixes replace the original single-TTS-anchor + hardcoded-L2-threshold design:

1. **Multi-voice prototype anchor (vs. the domain gap).** `keyword_generator.py` now synthesizes the keyword with ~30 voices (the 7 CMU ARCTIC speakers, random utterance-level x-vectors, and cross-speaker x-vector blends), "humanizes" each clip with random noise / synthetic room reverb / pitch / tempo / band-limit augmentation, embeds everything, and averages the L2-normalized embeddings into a **centroid anchor** (`keywords/<kw>_anchor.npz`). Single-voice and synthetic-domain idiosyncrasies partially average out; the shared phonetic content survives. Held-out voices are saved as calibration positives.

2. **Adaptive S-norm scoring (vs. the hardcoded threshold).** `cohort_builder.py` builds an impostor cohort (random keyword-length windows from the live stream + TTS distractor words, `cohort.npz`). `detector.py` scores each sliding window by cosine similarity on L2-normalized embeddings and normalizes it against the top-k closest cohort impostors on both the anchor side and the window side:

   `s_norm = 0.5 * ((s - mu_a)/sd_a + (s - mu_w)/sd_w)`

   The decision threshold lives in these calibrated units and is fitted per keyword by `calibrate.py` (target false-alarm point on the negative-score distribution: `mu_neg + sigma * sd_neg`). Accents, speaking pace, and room acoustics shift the anchor-vs-cohort and window-vs-cohort statistics together, so the normalized score stays comparable across conditions.

The detector scans at **multiple window scales** (default 0.6x/0.8x/1.0x of the TTS anchor duration) because conversational speech is often ~1.5-2x faster than SpeechT5's pace; a fixed TTS-length window dilutes the mean-pooled embedding with surrounding words. Note: the calibrated threshold is fitted on single-scale negatives, so multi-scale scanning raises the effective per-chunk false-alarm rate - use a stricter `--fa-percentile` (e.g. 99.8) if FPs appear.

`validate_detection.py` closes the loop: it transcribes each chunk with whisper-tiny and reports chunk-level precision/recall/F1 of the detector against the transcript ground truth. `transcribe_chunks.py` dumps all chunk transcripts to `transcripts.txt`.

**Known limitation (measured 2026-06-11):** detection quality is speaker/style-dependent with the current frozen model. On scripted news-anchor speech the TTS prototype anchor reached F1 0.57-0.80; on fast conversational debate speech it failed (the raw cosine of a confusable word, e.g. "appropriate" vs "penalty", exceeded that of the true keyword - an embedding-space ranking failure that no threshold can fix). This is the residual synthetic-to-real + anisotropy gap; the Phase-2 retrain (TTS-mixed triplets + gradient-reversal domain classifier) targets exactly this.

## Usage Guide

To run the pipeline from scratch on a new video:

1. **Download & Chunk:**
   ```bash
   python downloader.py
   ```
2. **VAD & Transcription (selects the keyword, also used for validation):**
   ```bash
   python transcriber.py
   ```
3. **Build the Multi-Voice Prototype Anchor:**
   ```bash
   python keyword_generator.py
   ```
4. **Build the AS-Norm Impostor Cohort:**
   ```bash
   python cohort_builder.py
   ```
5. **Calibrate the Per-Keyword Threshold:**
   ```bash
   python calibrate.py
   ```
6. **Run Live Zero-Shot Inference:**
   ```bash
   python detector.py
   ```
   (`--anchor-audio keywords/human_absolutely_real.m4a` overrides the TTS centroid with a recorded anchor.)
7. **Validate Against Whisper Ground Truth:**
   ```bash
   python validate_detection.py
   ```
