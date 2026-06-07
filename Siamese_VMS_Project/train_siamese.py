import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from siamese_model import SiameseAudioModel
from dataset import get_dataloader
import os

def train():
    print("Initializing Siamese Model Training on AWS GPU...")
    model = SiameseAudioModel()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Send only the projection head to training
    model.projection_head.train()
    
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.Adam(model.projection_head.parameters(), lr=1e-4)
    
    epochs = 50
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    dataloader = get_dataloader(batch_size=32)
    
    print(f"Starting {epochs} epochs of Production Training...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
            # The dataloader returns tensors. We convert them to list of numpy arrays for the HF feature extractor
            anchor_np = list(anchor.numpy())
            positive_np = list(positive.numpy())
            negative_np = list(negative.numpy())
            
            optimizer.zero_grad()
            
            # Forward pass through feature extractor + projection head
            anchor_embed = model.get_embedding(anchor_np, 16000)
            positive_embed = model.get_embedding(positive_np, 16000)
            negative_embed = model.get_embedding(negative_np, 16000)
            
            loss = criterion(anchor_embed, positive_embed, negative_embed)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx} | Loss: {loss.item():.4f}")
                
        avg_loss = epoch_loss / len(dataloader)
        print(f"--- Epoch {epoch+1} Average Loss: {avg_loss:.4f} ---")
        
        scheduler.step()
        
        # Save checkpoints every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"siamese_checkpoint_epoch_{epoch+1}.pth"
            model.save_weights(checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    print("Training complete. Saving final weights...")
    model.save_weights("siamese_finetuned.pth")
    print("Weights saved to siamese_finetuned.pth")

if __name__ == "__main__":
    train()
