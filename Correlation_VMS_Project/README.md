# Video Context Extraction

A comprehensive video stream analysis system that detects specific keywords in live video streams using synthesized speech and audio correlation detection. The project supports multiple parallel streams, integrates with YouTube Live streaming, and provides a GUI for easy monitoring and control.

## Project Overview

This project is designed to:
- Download and process live video streams from YouTube
- Split videos into manageable chunks
- Generate synthetic speech for search keywords using Microsoft SpeechT5 TTS (Text-To-Speech) with HiFi-GAN vocoder
- Perform cross-correlation analysis on audio streams to detect specified keywords
- Log detection events with timestamps
- Provide performance evaluation metrics (TP, FP, TN, FN)
- Support dual parallel streams (Stream 1 and Stream 2)
- Offer a GUI interface for stream management and monitoring

## Project Structure

### Core Components

#### 1. **Stream Processing Modules**

##### `Stream1_utube_vid_aud.py` / `Stream2_utube_vid_aud.py`
- **Purpose**: Download live video streams from YouTube URLs
- **Key Features**:
  - Accepts YouTube video URL as command-line argument
  - Uses `streamlink` to extract best quality stream segments
  - Parses M3U8 playlists to access stream chunks
  - Downloads video chunks with timestamps
  - Converts video to audio automatically using MoviePy
  - Multithreaded downloading for continuous stream capture
  - Handles network errors and retries automatically
- **Output**: MP4 video files and WAV audio files in respective directories

##### `Stream1_hifigan.py` / `Stream2_hifigan.py`
- **Purpose**: Generate synthetic speech for search keywords using AI models
- **Key Features**:
  - Uses Microsoft SpeechT5 (Transformer-based Text-To-Speech) model
  - Employs HiFi-GAN vocoder for high-quality audio generation
  - Supports multiple speaker profiles (Scottish, US, Canadian, Indian speakers)
  - Converts text input into natural-sounding speech
  - Outputs synthesized audio as WAV files
  - GPU acceleration support (CUDA)
- **Models Used**:
  - Text-To-Speech Model: `microsoft/speecht5_tts`
  - Vocoder: `microsoft/speecht5_hifigan`
  - Speaker Embeddings: `Matthijs/cmu-arctic-xvectors`

#### 2. **Audio Correlation Detection**

##### `Stream1_corelation_updated_v2.py` / `Stream2_corelation_updated_v2.py`
- **Purpose**: Detect keywords in audio streams using cross-correlation analysis
- **Key Features**:
  - Loads synthesized keyword audio files
  - Compares keyword audio against all streaming audio chunks
  - Performs NumPy cross-correlation to find matches
  - Implements threshold-based detection (70% matching percentage threshold)
  - Multi-threaded processing for efficiency
  - Synchronized detection events to stop processing once keyword found
  - Logs detection timestamps with exact video file reference
  - Extracts and saves matched audio segments
- **Output**: Timestamp logs showing detection events

#### 3. **Evaluation Module**

##### `evaluation.py`
- **Purpose**: Evaluate keyword detection performance and calculate metrics
- **Key Features**:
  - Splits video into 5-second chunks
  - Tracks True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN)
  - Computes detection accuracy based on known keyword timings
  - Provides performance statistics
  - Supports video preprocessing and segmentation
- **Metrics Output**: Displays TP, FP, TN, FN counts

#### 4. **GUI Application**

##### `vms.py` (Video Monitoring System)
- **Purpose**: Provide user-friendly interface for managing video streams
- **Key Features**:
  - Tkinter-based GUI interface
  - Support for multiple simultaneous streams
  - Input fields for search keywords (supports comma-separated lists)
  - Input fields for YouTube Live stream URLs
  - Start/Stop controls for stream processing
  - Real-time status updates
  - Displays creation timestamps and channel information
  - View button to open output folders
  - Video file counter for monitoring processed content
  - Frame-based layout for stream organization
- **Functionality**:
  - Enter search keywords or keyword lists
  - Provide live YouTube stream URLs
  - Toggle processing on/off
  - Monitor detection progress
  - Access saved video and audio files

## Directory Structure

The application uses the following directory structure automatically created in the user's home directory:

```
~/VMS/GUI2CHjetson/
├── Stream1videos/           # Downloaded Stream 1 video chunks
├── Stream1audios/           # Extracted audio from Stream 1 videos
├── Stream2videos/           # Downloaded Stream 2 video chunks
├── Stream2audios/           # Extracted audio from Stream 2 videos
├── Stream1_searchword1/     # Synthesized keyword audio for Stream 1
├── Stream2_searchword1/     # Synthesized keyword audio for Stream 2
└── Stream1_detection/       # Detection results and timestamps
    └── keyword/
        └── timestamps.txt   # Detection log
```

**Cross-Platform Compatibility**: All paths use `os.path.expanduser()` and `os.path.join()` for automatic Windows/Linux path handling.

## Dependencies

### Python Libraries
- `torch` - Deep learning framework for TTS model
- `transformers` - Hugging Face transformers for SpeechT5 and HiFi-GAN models
- `datasets` - Hugging Face datasets for speaker embeddings
- `librosa` - Audio analysis library
- `soundfile` - Audio file I/O
- `moviepy` - Video processing
- `streamlink` - Stream extraction
- `m3u8` - M3U8 playlist parsing
- `numpy` - Numerical computing
- `pytube` - YouTube video downloading
- `tkinter` - GUI framework (typically bundled with Python)

### External Tools
- FFmpeg - Video encoding/decoding (required by MoviePy)
- CUDA Toolkit - GPU acceleration (optional but recommended)

### Installation (All Platforms)

```bash
pip install torch transformers datasets librosa soundfile moviepy streamlink m3u8 numpy pytube
```

**Windows FFmpeg Installation:**
```bash
# Using chocolatey (recommended)
choco install ffmpeg

# Or download from: https://ffmpeg.org/download.html
```

**Linux FFmpeg Installation:**
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# Jetson
sudo apt-get install ffmpeg
```

## Workflow

### Standard Processing Pipeline

1. **Stream Acquisition**
   - User provides YouTube Live URL via GUI
   - `Stream*_utube_vid_aud.py` continuously downloads stream chunks
   - Videos automatically converted to audio

2. **Keyword Synthesis**
   - User enters search keywords in GUI
   - `Stream*_hifigan.py` generates synthetic speech for keywords
   - Audio saved as WAV files for comparison

3. **Keyword Detection**
   - `Stream*_corelation_updated_v2.py` loads synthesized keyword audio
   - Compares against incoming audio stream chunks
   - Cross-correlation detects matches above 70% threshold
   - Detection logged with timestamp and video segment reference

4. **Evaluation**
   - `evaluation.py` calculates performance metrics
   - Compares detected segments against ground truth
   - Reports TP, FP, TN, FN statistics

5. **Monitoring**
   - GUI updates with status, creation time, channel info
   - View button provides access to saved media
   - Video counter tracks processed content

## Usage Instructions

### GUI Mode (Recommended)
```bash
python vms.py
```

### Command-Line Mode

**Download Stream:**
```bash
python Stream1_utube_vid_aud.py <YOUTUBE_URL>
```

**Generate Keyword Audio:**
```bash
python Stream1_hifigan.py
```

**Run Detection:**
```bash
python Stream1_corelation_updated_v2.py
```

**Evaluate Results:**
```bash
python evaluation.py
```

## Configuration Notes

- **Cross-Platform Paths**: All scripts now use `os.path.expanduser()` and `os.path.join()` for automatic Windows/Linux/Jetson compatibility. Paths are automatically created in `~/VMS/GUI2CHjetson/`
- **GPU Support**: Scripts automatically detect CUDA availability and use GPU if available
- **Audio Format**: WAV files recommended for compatibility with librosa
- **Detection Threshold**: Currently set to 70% matching percentage for Stream1, 65% for Stream2 (adjustable in correlation scripts)
- **Chunk Duration**: Videos split into 5-second chunks for processing efficiency
- **Python Version**: Tested with Python 3.8+

## Platform Compatibility

✅ **Windows** (All versions)
- Paths automatically resolve to user home directory
- Create `~/VMS/GUI2CHjetson/` in user home

✅ **Linux/Jetson**
- Paths automatically resolve correctly
- Works in `/home/jetson/VMS/` or any Linux home directory

All path handling is now OS-agnostic and will work seamlessly across platforms without modification.

## Key Algorithms

### Cross-Correlation Detection
The detection mechanism uses NumPy's `correlate()` function to:
1. Load keyword audio signal
2. Compare against each stream audio chunk
3. Calculate matching percentage: `(max_correlation / (||match|| × ||keyword||)) × 100`
4. Trigger detection on threshold exceed (>70%)

### Threading Strategy
- Multiple threads handle concurrent downloads
- Event-based synchronization prevents redundant processing
- Local thread counters track processing state

## Performance Metrics

The system tracks:
- **TP (True Positives)**: Correctly detected keywords
- **FP (False Positives)**: Incorrectly detected keywords
- **TN (True Negatives)**: Correctly rejected non-keywords
- **FN (False Negatives)**: Missed keyword detections

## Supported Speakers (SpeechT5)

- `awb` - Scottish male
- `bdl` - US male
- `clb` - US female
- `jmk` - Canadian male
- `ksp` - Indian male
- `rms` - US male
- `slt` - US female

