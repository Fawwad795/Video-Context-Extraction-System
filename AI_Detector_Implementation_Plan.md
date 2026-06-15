# Implementation Plan — AI Keyword Detection for VMS (ASR-Based, Accuracy-First)

> **Audience:** Gemini 3.1 Pro (implementing agent).
> **Goal:** Replace the broken acoustic-Siamese detector (and the legacy `np.correlate` detector) with an **accurate, AI-based, open-vocabulary keyword detector** that localizes a user-typed keyword in live YouTube streams, timestamps it, and extracts the surrounding video context.
> **Decisions already made by the project owner:**
> - **Architecture:** Max-accuracy **ASR-based "transcribe-then-match"** (NOT a trainable acoustic-similarity / Siamese network).
> - **Deployment target:** PC / GPU / cloud (a GPU is available; large models are acceptable).
> - **Language:** English broadcast audio is the primary target (sports streams). Keep multilingual swappable but do not block on it.

---

## 0. TL;DR of the Strategy

The previous Siamese approach failed because it compared a **synthetic TTS robot voice** against **live human speech** using raw acoustic embeddings (mean-pooled `wav2vec2`). That matches *sound-wave topology*, not *language*. Documented result: F1 = 0.06–0.24 on real words (`evaluation_results.json`).

The fix — already validated implicitly inside the old repo — is to do **phonetic→text decoding with a large pretrained ASR (Whisper)**, then match the user's keyword against the **text transcript**. ASR is speaker-, accent-, and noise-robust because it was trained on 680k hours of human speech, and it eliminates the TTS-vs-human disparity entirely (there is no TTS anchor anymore).

> **Irony to exploit:** `transcriber.py` and `eval_generator.py` in `Siamese_VMS_Project/` **already use Whisper word-level timestamps** — they were used to *grade* the failing model. That same component **is** the new detector. We are promoting the grader to the product.

**New pipeline:**
```
YouTube live ─► chunk downloader ─► VAD (Silero) ─► WhisperX ASR (word-level timestamps)
   ─► rolling transcript w/ absolute stream time ─► keyword matcher (exact + phonetic fuzzy)
   ─► on match: log timestamp + extract video context (N chunks before/after) ─► GUI
```

---

## 1. Why This Fixes Every Documented Failure

| Retrospective Finding | Root cause | How ASR approach removes it |
|---|---|---|
| **F1: TTS Anchor Acoustic Disparity** | Compared TTS robot vs live human acoustically | No anchor at all. We decode audio→text, then compare *text to text*. Speaker/voice is irrelevant. |
| **F2: Threshold sensitivity / short-word failure** | L2 distance on raw embeddings; 1-syllable words swamped by crowd noise | Whisper decodes words using language-model context, not a brittle distance threshold. Confidence = word probability. |
| **F3: Anisotropy / cosine collapse** | Missing L2-norm in projection head; embeddings collapsed | No embedding-space metric used. Eliminated. |
| **Earlier Wav2Vec2 Urdu attempt: homophones / high WER** (Final Progress Report) | Small CTC model on low-resource language | English + Whisper large-v3 has dramatically lower WER; add phonetic fuzzy match + hotword biasing to absorb residual errors. |

**Expected outcome:** F1 should jump from ~0.2 to ~0.8–0.95 on clear English broadcast speech (Whisper large-v3 English WER is typically <5–10% on broadcast-quality audio).

---

## 2. Target Architecture (Detailed)

### 2.1 Core ASR engine
- **Primary:** [`WhisperX`](https://github.com/m-bain/whisperX) — Whisper + **wav2vec2 forced alignment** for accurate word-level start/end timestamps, plus VAD preprocessing (reduces hallucination), ~70× realtime batched on GPU.
- **Fallback / lighter option (config-swappable):** [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) (CTranslate2) with `word_timestamps=True`. Lower setup friction, supports `hotwords` keyword biasing and `int8` quantization.
- **Model size:** default `large-v3` on GPU; expose `MODEL_SIZE` in config so it can drop to `medium`/`small`/`base` for CPU or latency-bound runs.

> **Recommendation:** Implement an **`ASREngine` abstraction** with two backends (`whisperx`, `faster_whisper`) behind one interface so the rest of the system is backend-agnostic. Default to WhisperX for accuracy; allow `faster_whisper` for speed.

### 2.2 Keyword matcher (the new "detector" logic)
A keyword is detected when it appears in the transcript. Matching must tolerate ASR errors:
1. **Normalize** both keyword and transcript tokens: lowercase, strip punctuation, expand simple contractions, unicode-normalize.
2. **Multi-word phrases:** support keywords like `"red card"` by sliding an n-gram window over the token stream and matching the joined span; timestamp = start of first token → end of last token.
3. **Exact match** first (fast path).
4. **Phonetic fuzzy match** for near-misses: compute **Double Metaphone** (via `jellyfish` or `metaphone`) for keyword and each token; match if phonetic codes equal. This catches `"goal"` vs `"gaul"`-type ASR slips and accent-driven spellings.
5. **Edit-distance fallback:** normalized Levenshtein ratio (`rapidfuzz`) ≥ `FUZZY_RATIO` (default 0.85) as a secondary catch.
6. **Confidence:** combine the matched word's ASR probability (Whisper word `score`) with the match-type weight (exact=1.0, phonetic=0.9, fuzzy=ratio). Emit detection if `confidence ≥ CONF_THRESHOLD` (default 0.5; tune via eval sweep).
7. **Recall booster (optional but strong):** pass the user's keyword(s) to faster-whisper `hotwords=` / WhisperX `initial_prompt=` to bias the decoder toward producing the target word. Measure with and without; report the lift.

> **Why this beats a single threshold:** detection is now a discrete linguistic event (word present/absent) plus a calibrated confidence, not a fragile continuous distance.

### 2.3 Timestamp mapping (absolute stream time)
- Each chunk is `CHUNK_SECONDS` long (existing system uses ~5 s; live HLS segments may differ — read actual duration, don't assume).
- Maintain `chunk_start_time[chunk_index]` = cumulative duration of all prior chunks (sum real durations, do **not** multiply by a constant).
- Word absolute time = `chunk_start_time[chunk_index] + word_relative_time`.
- Store detections as `{keyword, chunk_index, abs_start_s, abs_end_s, wall_clock, confidence, match_type, transcript_context}`.

### 2.4 Context extraction (reuse + improve existing logic)
- On detection, save the surrounding video: `N_BEFORE` and `N_AFTER` chunks around the matched chunk (existing code uses 5 before / 5 after).
- **Reuse** the concatenation pattern from `Original VMS Project/Stream1_corelation_updated_v2.py::perform_concatenation`, BUT fix its flaw: concatenating raw `.mp4` byte streams via `shutil.copyfileobj` produces a technically-invalid container. **Use FFmpeg concat demuxer** (`ffmpeg -f concat -safe 0 -i list.txt -c copy out.mp4`) or MoviePy `concatenate_videoclips` for a valid output file.
- Write a `timestamps.txt` log per keyword (preserve existing format so the GUI's detection counter keeps working).

---

## 3. Repository Layout (new project)

Create a clean package; do not edit the old broken Siamese model files. Reuse infra (downloader, VAD, GUI shell) by copying/adapting.

```
ASR_VMS_Project/
├── config.py                 # all tunables: model size, backend, thresholds, chunk dirs, N_BEFORE/AFTER
├── asr_engine.py             # ASREngine abstraction: whisperx + faster_whisper backends
├── vad.py                    # Silero VAD wrapper (adapt from transcriber.py:has_human_speech)
├── keyword_matcher.py        # normalize + exact + phonetic(metaphone) + fuzzy(rapidfuzz) matching
├── stream_pipeline.py        # orchestrates: download → VAD → ASR → match → context-extract → log
├── downloader.py             # adapt Siamese_VMS_Project/downloader.py (continuous, not 10-chunk-capped)
├── context_extractor.py      # FFmpeg-based valid concatenation of N chunks around a hit
├── detector.py               # CLI entrypoint: run detection for a keyword over a live URL / folder
├── gui.py                    # adapt Original VMS Project/vms.py (Tkinter) to call the new pipeline
├── evaluate.py               # P/R/F1 harness vs Whisper-large ground truth (adapt eval_pipeline.py)
├── requirements.txt
└── README.md
```

---

## 4. File-by-File Implementation Spec

### 4.1 `config.py`
```python
import os
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
WORK_DIR   = os.path.join(BASE_DIR, "work")          # videos/, audios/, detections/, logs/
ASR_BACKEND = "whisperx"        # "whisperx" | "faster_whisper"
MODEL_SIZE  = "large-v3"        # large-v3 | medium | small | base | tiny
DEVICE      = "cuda"            # "cuda" | "cpu"
COMPUTE_TYPE = "float16"        # "float16" (GPU) | "int8" (CPU)
LANGUAGE    = "en"
CHUNK_SECONDS_HINT = 5.0        # only a hint; always read real duration from the file
N_BEFORE, N_AFTER  = 5, 5
CONF_THRESHOLD = 0.50
FUZZY_RATIO    = 0.85
USE_HOTWORD_BIAS = True
```

### 4.2 `asr_engine.py`
- Class `ASREngine(backend, model_size, device, compute_type, language)`.
- Method `transcribe_words(audio_path, hotwords=None) -> list[dict]` returning
  `[{"word": str, "start": float, "end": float, "score": float}, ...]` (relative to the file).
- **WhisperX path:** load `whisperx.load_model(...)`, run VAD-batched transcribe, then `whisperx.align(...)` with the language alignment model to get word timestamps. Cache the alignment model (load once).
- **faster-whisper path:** `WhisperModel(model_size, device, compute_type)`, `model.transcribe(audio, word_timestamps=True, hotwords=...)`; flatten `segment.words`.
- Normalize both backends to the same output schema. Load models **once** at construction (not per chunk).

### 4.3 `vad.py`
- Adapt `Siamese_VMS_Project/transcriber.py::has_human_speech` (Silero VAD via `torch.hub`, load with `librosa` to avoid torchaudio/torchcodec IO errors). Return speech timestamps so silent/music-only chunks are skipped before the expensive ASR call. (WhisperX has its own VAD; keep this as a cheap pre-gate and/or rely on WhisperX VAD — measure both.)

### 4.4 `keyword_matcher.py`
- `normalize(text) -> list[str]` tokens.
- `class KeywordMatcher(keywords: list[str], fuzzy_ratio, conf_threshold)`:
  - Precompute per-keyword: normalized tokens, n-gram length, Double Metaphone codes.
  - `find(words: list[dict]) -> list[Detection]` sliding the n-gram window; for each candidate span compute match_type (exact / phonetic / fuzzy) and `confidence = mean(word.score) * match_weight`; keep spans ≥ threshold.
  - Apply **non-max suppression** in time (reuse `eval_pipeline.py::non_max_suppression`) so one spoken word = one detection.
- Dependencies: `jellyfish` (metaphone) and `rapidfuzz` (edit distance).

### 4.5 `downloader.py`
- Adapt the existing Silver downloader but **run continuously** (the existing one caps at 10 chunks). Keep `streamlink` + `m3u8` segment fetch, dedupe by segment URI, convert mp4→wav with MoviePy. Write chunks as `live_{i}.mp4` / `live_{i}.wav` AND record each chunk's real duration to a sidecar (`durations.json`) for absolute-time mapping.
- Daemon thread + retry/backoff (already present in `Original VMS Project/Stream1_utube_vid_aud.py`).

### 4.6 `stream_pipeline.py`
- The orchestrator. For each newly downloaded chunk `i`:
  1. VAD gate → skip if no speech.
  2. `words = asr.transcribe_words(audio_path, hotwords=keywords if USE_HOTWORD_BIAS)`.
  3. Map word times to absolute using `durations.json`.
  4. `detections = matcher.find(words)`.
  5. For each detection: append to `logs/timestamps_{keyword}.txt` (legacy format `[time] Detected 'kw' in live_{i}.mp4`), and call `context_extractor.extract(i)`.
- Maintain a rolling transcript log for debugging/inspection.
- Process chunks as they arrive (queue). On GPU this is comfortably faster-than-realtime with `small`/`medium`; `large-v3` may need batching or a smaller model for true real-time — expose the trade-off in config.

### 4.7 `context_extractor.py`
- `extract(center_index)`: build list of `live_{center-N_BEFORE..center+N_AFTER}.mp4` that exist, concat **with FFmpeg concat demuxer** into `detections/{keyword}/{keyword}_detected_at_{center}.mp4`. Do not byte-concat. Clean up old chunks safely (reuse the `safe_remove` retry helper from `Stream1_corelation_updated_v2.py`).

### 4.8 `detector.py` (CLI)
```
python detector.py --url <youtube_live_url> --keywords "goal, red card, penalty"
python detector.py --audio-dir <folder> --keywords "goal"     # offline mode over existing chunks
```

### 4.9 `gui.py`
- Adapt `Original VMS Project/vms.py`. Keep the Tkinter dual-stream UI, Start/Stop, keyword entry, View folder, detection counter. Replace the subprocess calls to `*_hifigan.py` / `*_corelation_updated_v2.py` with calls into `stream_pipeline`. Remove the TTS keyword-synthesis step entirely (no longer needed — there is no audio anchor).

---

## 5. Evaluation Plan (prove the accuracy gain)

Reuse and adapt `Siamese_VMS_Project/eval_pipeline.py` + `eval_generator.py`:
1. **Ground truth:** transcribe the held-out eval chunks with **Whisper large-v3 word timestamps** (stronger than the `whisper-tiny` used before) → `eval_ground_truth.json`. Optionally hand-correct a small subset for a gold set.
2. **System under test:** run the new ASR detector over the same chunks for the same test words (reuse `["thank","sport","play","passion","level"]` plus a few multi-word phrases).
3. **Metrics:** per-keyword Precision / Recall / F1 with a time tolerance (±0.5 s, reuse `evaluate_predictions`). Report a single table + bar chart (reuse `eval_visualizer.py`).
4. **Ablations to include in the report:**
   - WhisperX vs faster-whisper.
   - With vs without hotword biasing.
   - Exact-only vs +phonetic vs +fuzzy matching.
   - Model size sweep (tiny→large-v3) — accuracy vs latency.
5. **Success criterion:** mean F1 ≥ 0.8 on clear English broadcast words (vs the old 0.06–0.24). Document any homophone failures explicitly.

> Keep the OLD `evaluation_results.json` and `pr_curves.png` as the "before" baseline for a compelling before/after comparison in the NESCOM extension write-up.

---

## 6. Dependencies (`requirements.txt`)

```
whisperx                # primary ASR + word alignment (pulls torch, faster-whisper, pyannote)
faster-whisper          # fallback backend / hotwords
torch                   # CUDA build matching the GPU
librosa, soundfile      # audio IO (reuse from existing project)
moviepy, imageio-ffmpeg # video chunking / conversion
streamlink, m3u8        # live stream capture
jellyfish               # Double Metaphone phonetic matching
rapidfuzz               # edit-distance fuzzy matching
numpy
```
- **FFmpeg** must be on PATH (already a project requirement).
- Note CUDA/cuDNN compatibility for WhisperX's wav2vec2 alignment + pyannote VAD.

---

## 7. Build Order (milestones for the implementing agent)

1. **M1 — ASR core:** `config.py` + `asr_engine.py`; unit-test `transcribe_words` on one existing `Siamese_VMS_Project/eval_audios/eval_*.wav`, confirm word + timestamp + score output.
2. **M2 — Matcher:** `keyword_matcher.py`; unit-test exact/phonetic/fuzzy/multi-word on synthetic token lists.
3. **M3 — Offline detection:** `context_extractor.py` + `stream_pipeline.py` + `detector.py --audio-dir` over existing eval chunks. Verify timestamps + valid concatenated mp4 output.
4. **M4 — Evaluation:** adapt `evaluate.py`; produce before/after F1 table + chart. **Gate:** mean F1 ≥ 0.8.
5. **M5 — Live:** `downloader.py` continuous mode; end-to-end `detector.py --url` on a real YouTube live stream.
6. **M6 — GUI:** `gui.py` wired to the pipeline; remove TTS step; confirm detection counter + View button.
7. **M7 — Polish:** logging, config sweeps, README, ablation results for the report.

---

## 8. Pitfalls / Gotchas (must-read for implementer)

- **Do NOT reintroduce any TTS anchor or acoustic distance metric.** There is no `keyword_generator.py` / `hifigan` step anymore. Detection is text-based.
- **Real durations, not constants:** HLS live segments are not guaranteed to be exactly 5 s. Read each chunk's duration; mis-mapping breaks absolute timestamps.
- **Byte-concatenating mp4s is broken** (the old code did this). Use FFmpeg concat demuxer for a valid file.
- **Load models once.** Constructing WhisperX/faster-whisper per chunk will be catastrophically slow.
- **Whisper hallucination on silence/music** → keep VAD gating; WhisperX VAD helps too.
- **Homophones / proper nouns** are the residual failure mode (e.g., team names). Mitigate with hotword biasing + phonetic match; document remaining cases honestly in the report.
- **Latency vs accuracy:** `large-v3` may not be true real-time per 5 s chunk on a single GPU. If real-time matters, default to `medium`/`small` and note the F1 delta; otherwise batch chunks.
- **Windows paths:** use `os.path.join` / `expanduser` throughout (the codebase already does); avoid the hardcoded `/home/jetson/...` paths that linger in `Stream2_corelation_updated_v2.py`.

---

## 9. What to Reuse vs Discard

**Reuse (adapt):**
- `Siamese_VMS_Project/downloader.py` (stream capture) → make continuous.
- `Siamese_VMS_Project/transcriber.py::has_human_speech` (Silero VAD).
- `Siamese_VMS_Project/eval_pipeline.py` (`non_max_suppression`, `evaluate_predictions`) + `eval_generator.py` (Whisper word-timestamp GT) + `eval_visualizer.py`.
- `Original VMS Project/vms.py` (Tkinter GUI shell) + the `safe_remove` / concatenation *structure* (not the byte-concat) from the correlation scripts.

**Discard (do not carry forward):**
- `siamese_model.py`, `detector.py` (Siamese), `train_siamese.py`, `dataset.py`, `dtw_test.py`, all `*hifigan*`, `keyword_generator.py`, `*.pth` weights — the entire acoustic-similarity path.
- All `np.correlate` detection logic from the Original VMS Project.

---

## 10. One-paragraph summary for the report / commit message

> The keyword detector was re-architected from acoustic-similarity matching (a Siamese `wav2vec2` network comparing a synthetic TTS anchor against live audio, which failed at F1 ≈ 0.06–0.24 due to TTS-vs-human acoustic disparity) to a **phonetic, ASR-based transcribe-then-match design**. Live stream chunks are gated by Silero VAD, transcribed with WhisperX word-level forced alignment, and the user's keyword is matched against the transcript via exact + Double-Metaphone phonetic + edit-distance fuzzy matching with hotword biasing. This is speaker-, accent-, and noise-robust by construction, open-vocabulary, and yields exact word timestamps for context extraction — fixing all three failure modes documented in the project retrospective.
