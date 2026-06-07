import os
import torch
import soundfile as sf
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import librosa

def generate_eval_keywords():
    test_words = [
        "thank", "sport", "think", "team", "girls", 
        "level", "passion", "moment", "timmy", "after", 
        "make", "play", "goal", "year", "just"
    ]
    
    base_path = r"d:\Video Context Extraction System\Siamese_VMS_Project"
    output_dir = os.path.join(base_path, "eval_keywords")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading SpeechT5 models...")
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    
    print("Loading speaker embeddings...")
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    
    for word in test_words:
        output_path = os.path.join(output_dir, f"{word}.wav")
        if os.path.exists(output_path):
            print(f"Skipping {word} (already exists)")
            continue
            
        print(f"Synthesizing '{word}'...")
        inputs = processor(text=word, return_tensors="pt")
        
        with torch.no_grad():
            speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
            
        # Strip silence
        speech_np = speech.numpy()
        trimmed_speech, _ = librosa.effects.trim(speech_np, top_db=30)
        
        sf.write(output_path, trimmed_speech, samplerate=16000)
        
    print("Finished generating all 15 test keywords.")

if __name__ == "__main__":
    generate_eval_keywords()
