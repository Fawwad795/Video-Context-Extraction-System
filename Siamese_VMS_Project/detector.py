import os
import glob
import librosa
from datetime import datetime
from siamese_model import SiameseAudioModel

def run_detection(keyword_audio_path, live_audio_dir, output_log_path, threshold=0.75):
    print("Initializing Siamese AI Detector...")
    model = SiameseAudioModel()
    
    best_model_path = os.path.join(os.path.dirname(keyword_audio_path), "..", "best_siamese_model.pth")
    if os.path.exists(best_model_path):
        print(f"Loading optimal AWS checkpoint: {best_model_path}")
        model.load_weights(best_model_path)
        # Since the network trained on 5 million examples and converged to a validation loss of ~0.3,
        # identical words are mapped incredibly tightly. However, to account for acoustic variance across
        # different real-world speakers, we will use a threshold of 0.75 to prevent false-negatives.
        threshold = 0.75
    else:
        print("Using raw zero-shot embeddings. Threshold might need adjusting.")
        
    # Load keyword audio
    print(f"Loading keyword audio from: {keyword_audio_path}")
    keyword_y_raw, keyword_sr = librosa.load(keyword_audio_path, sr=16000)
    
    # Strip leading/trailing silence from the synthetic TTS to ensure sliding window is exactly the word length
    keyword_y, _ = librosa.effects.trim(keyword_y_raw, top_db=30)
    
    # Get reference embedding
    print("Generating reference embedding for keyword...")
    keyword_embed = model.get_embedding(keyword_y, keyword_sr)
    
    # Keyword duration
    keyword_duration = len(keyword_y) / keyword_sr
    
    # Get live audio chunks
    audio_files = glob.glob(os.path.join(live_audio_dir, "*.wav"))
    audio_files.sort(key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]))
    
    os.makedirs(os.path.dirname(output_log_path), exist_ok=True)
    
    print(f"\nStarting detection on {len(audio_files)} live chunks. Threshold: {threshold}")
    
    for audio_file in audio_files:
        filename = os.path.basename(audio_file)
        y, sr = librosa.load(audio_file, sr=16000)
        
        # Sliding window parameters
        window_size = len(keyword_y)  # window size matches keyword length
        step_size = int(sr * 0.1)     # 0.1 second steps for fine granularity
        
        chunk_detected = False
        min_distance = float('inf')
        
        for i in range(0, len(y) - window_size, step_size):
            window_audio = y[i:i + window_size]
            
            # Get embedding for the current sliding window
            window_embed = model.get_embedding(window_audio, sr)
            
            # Compute Siamese L2 Distance
            distance = model.compute_similarity(keyword_embed, window_embed)
            if distance < min_distance:
                min_distance = distance
            
            if distance < threshold:
                timestamp_seconds = i / sr
                detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                msg = f"[{detection_time}] AI Match Found! L2 Distance: {distance:.2f} | Chunk: {filename} at {timestamp_seconds:.1f}s"
                print(msg)
                
                with open(output_log_path, "a") as f:
                    f.write(msg + "\n")
                    
                chunk_detected = True
                break # Move to next chunk after first detection to avoid spam
                
        if not chunk_detected:
            print(f"Processed {filename} - No matches found < {threshold}. (Closest distance: {min_distance:.2f})")

if __name__ == "__main__":
    base_path = r"d:\Video Context Extraction System\Siamese_VMS_Project"
    
    # Locate keyword
    keyword_file = os.path.join(base_path, "selected_keyword.txt")
    if not os.path.exists(keyword_file):
        print("Keyword file not found. Run transcriber.py first.")
        exit()
        
    with open(keyword_file, "r") as f:
        keyword = f.read().strip()
        
    keyword_audio_path = os.path.join(base_path, "keywords", f"{keyword}.wav")
    live_audio_dir = os.path.join(base_path, "audios")
    output_log_path = os.path.join(base_path, "logs", f"timestamps_{keyword}.txt")
    
    if not os.path.exists(keyword_audio_path):
        print(f"Keyword audio {keyword_audio_path} not found. Run keyword_generator.py first.")
        exit()
        
    run_detection(keyword_audio_path, live_audio_dir, output_log_path, threshold=0.75)
