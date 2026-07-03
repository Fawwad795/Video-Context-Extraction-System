# hifigan_updated.py

# Import necessary libraries
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
import os
import argparse  # New import for command-line argument parsing

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the processor
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")

# Load the model
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)

# Load the vocoder (voice encoder)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

# Load the dataset to get speaker embeddings
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

# Speaker ids from the embeddings dataset
speakers = {
    'awb': 0,     # Scottish male
    'bdl': 1138,  # US male
    'clb': 2271,  # US female
    'jmk': 3403,  # Canadian male
    'ksp': 4535,  # Indian male
    'rms': 5667,  # US male
    'slt': 6799   # US female
}

def save_text_to_speech(text, speaker=None):
    # Preprocess text
    inputs = processor(text=text, return_tensors="pt").to(device)

    if speaker is not None:
        # Load xvector containing speaker's voice characteristics from a dataset
        speaker_embeddings = torch.tensor(embeddings_dataset[speaker]["xvector"]).unsqueeze(0).to(device)
    else:
        # Random vector, meaning a random voice
        speaker_embeddings = torch.randn((1, 512)).to(device)

    # Generate speech with the models
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

    if speaker is not None:
        # If we have a speaker, we use the speaker's ID in the filename
        base_path = os.path.expanduser(os.path.join('~', 'VMS', 'GUI2CHjetson'))
        output_filedirectory = os.path.join(base_path, 'Stream1_searchword1')
        output_filename = os.path.join(output_filedirectory, f"{speaker}-{'-'.join(text.split()[:6])}.wav")

        # Create directories if they don't exist
        if not os.path.isdir(output_filedirectory):
            os.makedirs(output_filedirectory)
            print('Directory Created for Synthesized file')

    # Save the generated speech to a file with a 16KHz sampling rate
    sf.write(output_filename, speech.cpu().numpy(), samplerate=16000)

    # Return the filename for reference
    return output_filename

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate audio from a word using HiFIGAN.")
    parser.add_argument("word", type=str, help="The word for audio generation.")
    args = parser.parse_args()

    # Process the word and generate audio
    text = args.word
    print("Entered word is:", text)

    for value in speakers:
        output_filename = save_text_to_speech(text, speaker=speakers[value])
        print(f"Processing value: {value}")

