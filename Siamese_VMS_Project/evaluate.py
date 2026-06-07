import torch
import torch.nn as nn
from datasets import load_dataset
import numpy as np
import random
import librosa
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from siamese_model import SiameseAudioModel
import os
import glob

class ValidationTripletDataset(Dataset):
    def __init__(self, num_samples=1000):
        print("Loading MLCommons/ml_spoken_words validation split...")
        self.dataset = load_dataset("MLCommons/ml_spoken_words", "en_wav", split="validation", trust_remote_code=True)
        
        self.class_to_indices = defaultdict(list)
        
        all_keywords = self.dataset["keyword"]
        for idx, label in enumerate(all_keywords):
            if label:
                self.class_to_indices[label].append(idx)
                
        self.classes = list(self.class_to_indices.keys())
        self.classes = [c for c in self.classes if len(self.class_to_indices[c]) >= 3]
        
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples
        
    def _get_audio_array(self, idx):
        item = self.dataset[idx]
        audio_array = item["audio"]["array"]
        sr = item["audio"]["sampling_rate"]
        
        if sr != 16000:
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
            
        if len(audio_array) > 16000:
            audio_array = audio_array[:16000]
        else:
            audio_array = np.pad(audio_array, (0, 16000 - len(audio_array)))
            
        return audio_array.astype(np.float32)

    def __getitem__(self, idx):
        anchor_class = random.choice(self.classes)
        negative_class = random.choice(self.classes)
        while negative_class == anchor_class:
            negative_class = random.choice(self.classes)
            
        anchor_idx, positive_idx = random.sample(self.class_to_indices[anchor_class], 2)
        negative_idx = random.choice(self.class_to_indices[negative_class])
        
        anchor = self._get_audio_array(anchor_idx)
        positive = self._get_audio_array(positive_idx)
        negative = self._get_audio_array(negative_idx)
        
        return anchor, positive, negative

def evaluate():
    print("Initializing Siamese Evaluation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    val_dataset = ValidationTripletDataset(num_samples=500)
    dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    
    checkpoints = glob.glob("siamese_checkpoint_epoch_*.pth")
    checkpoints.append("siamese_finetuned.pth")
    
    # Sort checkpoints by epoch number (or put finetuned at end)
    def extract_epoch(ckpt):
        if "finetuned" in ckpt: return 999
        try: return int(ckpt.split("_epoch_")[1].split(".")[0])
        except: return 0
    
    checkpoints.sort(key=extract_epoch)
    
    best_loss = float('inf')
    best_checkpoint = None

    for checkpoint in checkpoints:
        if not os.path.exists(checkpoint):
            continue
            
        print(f"\nEvaluating Checkpoint: {checkpoint}")
        model = SiameseAudioModel()
        model.to(device)
        model.eval() # Set to eval mode
        
        try:
            model.load_weights(checkpoint)
        except Exception as e:
            print(f"Skipping {checkpoint} - error loading weights: {e}")
            continue

        total_loss = 0.0
        with torch.no_grad():
            for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
                anchor_np = list(anchor.numpy())
                positive_np = list(positive.numpy())
                negative_np = list(negative.numpy())
                
                anchor_embed = model.get_embedding(anchor_np, 16000)
                positive_embed = model.get_embedding(positive_np, 16000)
                negative_embed = model.get_embedding(negative_np, 16000)
                
                loss = criterion(anchor_embed, positive_embed, negative_embed)
                total_loss += loss.item()
                
        avg_loss = total_loss / len(dataloader)
        print(f"Validation Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_checkpoint = checkpoint

    print(f"\n=====================================")
    print(f"EVALUATION COMPLETE")
    print(f"Best Checkpoint: {best_checkpoint}")
    print(f"Best Validation Loss: {best_loss:.4f}")
    print(f"=====================================")
    
    # Symlink the best to 'best_siamese_model.pth'
    if best_checkpoint:
        os.system(f"cp {best_checkpoint} best_siamese_model.pth")
        print(f"Copied {best_checkpoint} to best_siamese_model.pth for production use.")

if __name__ == "__main__":
    evaluate()
