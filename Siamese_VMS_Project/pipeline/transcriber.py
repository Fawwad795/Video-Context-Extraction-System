import os
import glob
from collections import Counter
from transformers import pipeline
import torch
import warnings

# Suppress PyTorch warnings for clean output
warnings.filterwarnings("ignore")

def has_human_speech(audio_path):
    """
    Uses Silero VAD to check if an audio file contains human speech.
    Returns True if speech is found, False otherwise.
    """
    try:
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=False,
                                      trust_repo=True)
        (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
        
        # Use librosa to avoid torchaudio IO torchcodec dependency error
        import librosa
        wav_np, _ = librosa.load(audio_path, sr=16000)
        wav = torch.from_numpy(wav_np)
        
        speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)
        
        return len(speech_timestamps) > 0
    except Exception as e:
        print(f"  VAD Error: {e}")
        return True # Fallback to True if VAD fails so we still attempt transcription

def transcribe_and_find_keyword(audio_dir):
    print("Loading whisper model for transcription...")
    # Using whisper-tiny for speed
    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
    
    audio_files = glob.glob(os.path.join(audio_dir, "*.wav"))
    if not audio_files:
        print("No audio files found.")
        return None
        
    all_words = []
    
    # Common English stop words
    stop_words = set(["the", "be", "to", "of", "and", "a", "in", "that", "have", "i", 
                      "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
                      "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
                      "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
                      "so", "up", "out", "if", "about", "who", "get", "which", "go", "me", "is", 
                      "are", "was", "were", "been", "has", "had", "can", "could", "should"])

    print(f"Transcribing {len(audio_files)} audio chunks...")
    for idx, audio_path in enumerate(audio_files):
        print(f"[{idx+1}/{len(audio_files)}] Processing {os.path.basename(audio_path)}...")
        
        # 1. Run Voice Activity Detection (VAD)
        if not has_human_speech(audio_path):
            print("  Skipped: No human speech detected (likely music or silence).")
            continue
            
        # 2. Transcribe only if speech is found
        try:
            result = transcriber(audio_path)
            text = result["text"].lower()
            print(f"  Transcript: {text}")
            
            # Simple tokenization
            import re
            words = re.findall(r'\b[a-z]{3,}\b', text)
            
            # Filter stop words
            meaningful_words = [w for w in words if w not in stop_words]
            all_words.extend(meaningful_words)
        except Exception as e:
            print(f"  Error transcribing: {e}")

    if not all_words:
        print("No meaningful words found.")
        return None
        
    # Count frequencies
    word_counts = Counter(all_words)
    most_common = word_counts.most_common(5)
    print("\nMost frequent words:")
    for word, count in most_common:
        print(f"  {word}: {count} times")
        
    top_word = most_common[0][0]
    print(f"\nSelected keyword for detection: '{top_word}'")
    
    # Save the keyword to a file so other scripts can use it
    with open(os.path.join(base_path, "selected_keyword.txt"), "w") as f:
        f.write(top_word)
        
    return top_word

if __name__ == "__main__":
    base_path = r"d:\Video Context Extraction System\Siamese_VMS_Project"
    audio_dir = os.path.join(base_path, "audios")
    transcribe_and_find_keyword(audio_dir)
