import os
import json
import re
from collections import Counter
import torch
from transformers import pipeline

def clean_word(word):
    # Remove punctuation, whitespace, and convert to lowercase
    cleaned = re.sub(r'[^\w\s]', '', word).strip().lower()
    return cleaned

def generate_ground_truth(audio_dir, output_json_path, num_chunks=60):
    print("Loading whisper model for word-level transcription...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transcriber = pipeline(
        "automatic-speech-recognition", 
        model="openai/whisper-tiny",
        device=device,
        return_timestamps="word"
    )

    ground_truth = {}
    word_counter = Counter()

    for i in range(num_chunks):
        audio_file = os.path.join(audio_dir, f"eval_{i}.wav")
        if not os.path.exists(audio_file):
            print(f"Skipping {audio_file} (not found)")
            continue

        print(f"[{i+1}/{num_chunks}] Transcribing eval_{i}.wav...")
        try:
            result = transcriber(audio_file)
            chunks = result.get("chunks", [])
            
            file_gt = []
            for chunk in chunks:
                raw_word = chunk.get("text", "")
                word = clean_word(raw_word)
                timestamp = chunk.get("timestamp", (None, None))
                
                if word and timestamp[0] is not None and timestamp[1] is not None:
                    file_gt.append({
                        "word": word,
                        "start": timestamp[0],
                        "end": timestamp[1]
                    })
                    word_counter[word] += 1
            
            ground_truth[f"eval_{i}.wav"] = file_gt
            
        except Exception as e:
            print(f"Error transcribing {audio_file}: {e}")

    with open(output_json_path, "w") as f:
        json.dump(ground_truth, f, indent=4)
    print(f"Ground truth saved to {output_json_path}")

    # Exclude common stop words
    stop_words = {"the", "a", "and", "is", "to", "in", "of", "it", "that", "you", "for", "on", "with", "as", "at"}
    
    print("\n--- Top 30 Most Frequent Words (Excluding Stop Words) ---")
    valid_words = [(w, c) for w, c in word_counter.most_common() if w not in stop_words and len(w) > 2]
    for w, c in valid_words[:30]:
        print(f"  {w}: {c} times")

if __name__ == "__main__":
    base_path = r"d:\Video Context Extraction System\Siamese_VMS_Project"
    audio_dir = os.path.join(base_path, "eval_audios")
    output_json = os.path.join(base_path, "eval_ground_truth.json")
    
    generate_ground_truth(audio_dir, output_json, num_chunks=60)
