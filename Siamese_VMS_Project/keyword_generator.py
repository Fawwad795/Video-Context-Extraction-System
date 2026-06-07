import os
import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import soundfile as sf

def generate_keyword_audio(keyword, output_dir):
    print(f"Generating synthesized audio for keyword: '{keyword}'")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"{keyword}.wav")
    
    # Load models
    print("Loading SpeechT5 models...")
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    
    # Process text
    inputs = processor(text=keyword, return_tensors="pt")
    
    # Load speaker embeddings (using standard dataset)
    print("Loading speaker embeddings...")
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    
    # Generate audio
    print("Synthesizing speech...")
    with torch.no_grad():
        speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
        
    # Save audio
    sf.write(output_path, speech.numpy(), samplerate=16000)
    print(f"Keyword audio saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    base_path = r"d:\Video Context Extraction System\Siamese_VMS_Project"
    keyword_file = os.path.join(base_path, "selected_keyword.txt")
    
    if not os.path.exists(keyword_file):
        print(f"Error: {keyword_file} not found. Run transcriber.py first.")
    else:
        with open(keyword_file, "r") as f:
            keyword = f.read().strip()
        
        output_dir = os.path.join(base_path, "keywords")
        generate_keyword_audio(keyword, output_dir)
