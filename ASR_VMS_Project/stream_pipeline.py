import os
import json
import time
from datetime import datetime
from config import WORK_DIR, ASR_BACKEND, MODEL_SIZE, DEVICE, COMPUTE_TYPE, LANGUAGE, CONF_THRESHOLD, FUZZY_RATIO, USE_HOTWORD_BIAS
from asr_engine import ASREngine
from vad import VAD
from keyword_matcher import KeywordMatcher
from context_extractor import extract

class StreamPipeline:
    def __init__(self, keywords):
        self.keywords = keywords
        print("Initializing ASR Engine...")
        self.asr = ASREngine(backend=ASR_BACKEND, model_size=MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE, language=LANGUAGE)
        print("Initializing VAD...")
        self.vad = VAD(device=DEVICE)
        print("Initializing Keyword Matcher...")
        self.matcher = KeywordMatcher(keywords, fuzzy_ratio=FUZZY_RATIO, conf_threshold=CONF_THRESHOLD)
        
        # Load known durations if any
        self.durations_file = os.path.join(WORK_DIR, "durations.json")
        self.durations = {}
        if os.path.exists(self.durations_file):
            with open(self.durations_file, "r") as f:
                self.durations = json.load(f)
                
    def process_chunk(self, chunk_index, audio_path, live_prefix="live"):
        if not os.path.exists(audio_path):
            return
            
        # VAD Gate
        if not self.vad.has_speech(audio_path):
            print(f"[{live_prefix}_{chunk_index}] Skipped (no speech detected)")
            return
            
        # ASR
        print(f"[{live_prefix}_{chunk_index}] Transcribing...")
        hotwords = ", ".join(self.keywords) if USE_HOTWORD_BIAS else None
        
        # ASR could throw errors on bad files, catch them
        try:
            words = self.asr.transcribe_words(audio_path, hotwords=hotwords)
        except Exception as e:
            print(f"[{live_prefix}_{chunk_index}] Transcription failed: {e}")
            return
            
        if not words:
            return
            
        # Time mapping
        chunk_key = f"{live_prefix}_{chunk_index}"
        chunk_duration = self.durations.get(chunk_key, 5.0) # default to 5s if unknown
        
        # We need absolute time = sum of previous durations
        abs_start_time = 0.0
        for i in range(chunk_index):
            prev_key = f"{live_prefix}_{i}"
            abs_start_time += self.durations.get(prev_key, 5.0)
            
        # Match
        detections = self.matcher.find(words)
        
        for d in detections:
            abs_det_start = abs_start_time + d["start"]
            abs_det_end = abs_start_time + d["end"]
            wall_clock = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            msg = f"[{wall_clock}] Detected '{d['keyword']}' (type: {d['match_type']}, conf: {d['confidence']:.2f}) at stream time {abs_det_start:.1f}s (chunk {chunk_index})"
            print(">>>", msg)
            
            # Log to file
            log_file = os.path.join(WORK_DIR, "logs", f"timestamps_{d['keyword']}.txt")
            with open(log_file, "a") as f:
                f.write(msg + "\n")
                
            # Extract Context
            extract(chunk_index, d['keyword'], live_prefix)
