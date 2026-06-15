import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = os.path.join(BASE_DIR, "work")

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

# Create necessary directories
os.makedirs(os.path.join(WORK_DIR, "audios"), exist_ok=True)
os.makedirs(os.path.join(WORK_DIR, "videos"), exist_ok=True)
os.makedirs(os.path.join(WORK_DIR, "detections"), exist_ok=True)
os.makedirs(os.path.join(WORK_DIR, "logs"), exist_ok=True)
