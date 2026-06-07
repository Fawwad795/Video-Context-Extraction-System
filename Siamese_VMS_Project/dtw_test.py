import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import numpy as np

# A simple DTW implementation for PyTorch tensors (or numpy)
def compute_dtw(seq1, seq2):
    # seq1: (N, 768)
    # seq2: (M, 768)
    # Normalize features to prevent magnitude bias
    seq1 = torch.nn.functional.normalize(seq1, p=2, dim=1)
    seq2 = torch.nn.functional.normalize(seq2, p=2, dim=1)
    
    N, M = seq1.shape[0], seq2.shape[0]
    cost_matrix = torch.cdist(seq1, seq2, p=2) # (N, M) pairwise distances
    
    dtw = torch.full((N+1, M+1), float('inf'))
    dtw[0, 0] = 0
    
    for i in range(1, N+1):
        for j in range(1, M+1):
            cost = cost_matrix[i-1, j-1]
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
            
    # Return normalized DTW distance
    return (dtw[N, M] / max(N, M)).item()

print("Loading raw Wav2Vec2 Backbone...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
model.eval()

def get_frames(audio_path, start_time=None, duration=None):
    y, sr = librosa.load(audio_path, sr=16000, offset=start_time, duration=duration)
    inputs = processor(y, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        features = model(**inputs).last_hidden_state[0] # (Seq_Len, 768)
    return features

print("Extracting frames...")
anchor_frames = get_frames("keywords/human_absolutely.wav")

# Extract the TRUE match (live_5 at 3.5s)
true_match_frames = get_frames("audios/live_5.wav", start_time=3.5, duration=1.0)

# Extract the FALSE positive match (live_2 at 2.0s) from earlier test
false_match_frames = get_frames("audios/live_2.wav", start_time=2.0, duration=1.0)

dist_true = compute_dtw(anchor_frames, true_match_frames)
dist_false = compute_dtw(anchor_frames, false_match_frames)

print(f"DTW Distance to True Match (live_5): {dist_true:.4f}")
print(f"DTW Distance to False Positive (live_2): {dist_false:.4f}")
