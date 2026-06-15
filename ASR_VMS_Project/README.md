# ASR Video Monitoring System (ASR_VMS_Project)

An accurate, AI-based, open-vocabulary keyword detector that localizes a user-typed keyword in live streams, timestamps it, and extracts the surrounding video context.

## Architecture

This project uses an **ASR-based "transcribe-then-match"** approach, replacing older acoustic-similarity matching techniques. It processes live streams by:

1. **VAD (Voice Activity Detection)**: Uses Silero VAD to skip silence.
2. **ASR (Automatic Speech Recognition)**: Uses `whisperx` (or `faster-whisper`) to decode human speech into text with precise word-level timestamps.
3. **Keyword Matching**: Searches the generated transcript using exact match, Double Metaphone (phonetic) matching, and Levenshtein (fuzzy edit distance) matching.
4. **Context Extraction**: Upon finding a match, the system uses FFmpeg's concat demuxer to flawlessly stitch the adjacent video chunks into a single context video clip.

## Setup

1. Create a virtual environment: `python -m venv venv`
2. Activate it: `.\venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Linux/Mac)
3. Install dependencies: `pip install -r requirements.txt`

## Usage

### GUI Mode
The easiest way to use the system is via the Tkinter interface:
```bash
python gui.py
```

### CLI Mode
You can also run detections from the command line on offline audio chunk datasets:
```bash
python detector.py --audio-dir ../Siamese_VMS_Project/eval_audios --keywords "goal, red card, penalty"
```

## Configuration

Modify `config.py` to tune the application:
- `ASR_BACKEND`: Choose between `whisperx` and `faster_whisper`
- `MODEL_SIZE`: Whisper model size (`large-v3`, `medium`, `small`, `base`, `tiny`)
- `CONF_THRESHOLD`: Minimum confidence (0.0 to 1.0)
- `FUZZY_RATIO`: Levenshtein ratio threshold for fuzzy text matching (default 85.0)
