import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

class SiameseAudioModel(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base", embed_dim=128):
        super(SiameseAudioModel, self).__init__()
        print(f"Loading pre-trained backbone ({model_name}) for Siamese Network...")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.backbone = Wav2Vec2Model.from_pretrained(model_name, use_safetensors=True)
        
        # We freeze the backbone to act as a pure feature extractor
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Add a trainable projection head for Metric Learning
        # This will be fine-tuned using Triplet Loss
        self.projection_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )
            
    def get_embedding(self, audio_array, sampling_rate=16000):
        """
        Extracts a fixed-length embedding vector for a given audio segment.
        """
        # Ensure audio is at 16kHz
        if sampling_rate != 16000:
            audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)
            
        # Process audio to model input format
        inputs = self.feature_extractor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True)
        
        # Move inputs to the same device as the model
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Extract the last hidden state: [Batch, Time, Features]
            outputs = self.backbone(**inputs)
            hidden_states = outputs.last_hidden_state
        
        # Mean pooling across the time dimension
        embedding = torch.mean(hidden_states, dim=1)  # [Batch, Features (768)]
        
        # Pass through trainable projection head
        projected = self.projection_head(embedding) # [Batch, 128]
        return projected
        
    def save_weights(self, path):
        torch.save(self.projection_head.state_dict(), path)
        print(f"Weights saved to {path}")
        
    def load_weights(self, path):
        self.projection_head.load_state_dict(torch.load(path, map_location="cpu"))
        self.projection_head.eval()
        print(f"Weights loaded from {path}")
        
    def compute_similarity(self, embed1, embed2):
        """
        Computes the Euclidean Distance (L2 norm) between two audio embeddings.
        Because we trained with TripletMarginLoss(p=2, margin=1.0),
        a distance < 1.0 indicates a strong match.
        """
        return torch.dist(embed1, embed2, p=2).item()

    def process_and_compare(self, audio1, sr1, audio2, sr2):
        """
        End-to-end function to compare two raw audio arrays.
        """
        emb1 = self.get_embedding(audio1, sr1)
        emb2 = self.get_embedding(audio2, sr2)
        return self.compute_similarity(emb1, emb2)
