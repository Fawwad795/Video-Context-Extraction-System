import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import numpy as np
import random
import librosa
from collections import defaultdict

class SpeechTripletDataset(Dataset):
    def __init__(self, num_samples=100000):
        print("Loading MLCommons/ml_spoken_words dataset for massive vocabulary triplet scaling...")
        
        # Load MLCommons/ml_spoken_words (hundreds of thousands of words)
        self.dataset = load_dataset("MLCommons/ml_spoken_words", "en_wav", split="train", trust_remote_code=True)
        
        self.class_to_indices = defaultdict(list)
        
        # Group samples by their word label using ONLY indices (memory efficient)
        print("Grouping dataset indices by keyword class...")
        all_keywords = self.dataset["keyword"]
        for idx, label in enumerate(all_keywords):
            if label:
                self.class_to_indices[label].append(idx)
                
        self.classes = list(self.class_to_indices.keys())
        # Filter classes that have at least 3 samples
        self.classes = [c for c in self.classes if len(self.class_to_indices[c]) >= 3]
        
        # The true number of possible triplets is immense. We just define an epoch size.
        self.num_samples = num_samples
        print(f"Dataset ready. Found {len(self.classes)} distinct keyword classes.")

    def __len__(self):
        return self.num_samples
        
    def _get_audio_array(self, idx):
        item = self.dataset[idx]
        audio_array = item["audio"]["array"]
        sr = item["audio"]["sampling_rate"]
        
        if sr != 16000:
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
            
        # speech_commands is typically exactly 1 second (16000 samples), but we pad/truncate just in case
        if len(audio_array) > 16000:
            audio_array = audio_array[:16000]
        else:
            audio_array = np.pad(audio_array, (0, 16000 - len(audio_array)))
            
        return audio_array.astype(np.float32)

    def __getitem__(self, idx):
        # 1. Pick a random anchor class
        anchor_class = random.choice(self.classes)
        
        # 2. Pick a negative class
        negative_class = random.choice(self.classes)
        while negative_class == anchor_class:
            negative_class = random.choice(self.classes)
            
        # 3. Pick Anchor and Positive indices (different speakers saying same word)
        anchor_idx, positive_idx = random.sample(self.class_to_indices[anchor_class], 2)
        
        # 4. Pick Negative index
        negative_idx = random.choice(self.class_to_indices[negative_class])
        
        # Lazy load the actual audio arrays from HuggingFace arrow tables
        anchor = self._get_audio_array(anchor_idx)
        positive = self._get_audio_array(positive_idx)
        negative = self._get_audio_array(negative_idx)
        
        return anchor, positive, negative

def get_dataloader(batch_size=32):
    # For production, we'll run 10,000 random triplets per epoch
    ds = SpeechTripletDataset(num_samples=10000)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)
