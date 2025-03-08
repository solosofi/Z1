"""
Training script for Onyx.

This file demonstrates a full training pipeline:
- Data generation (dummy synthetic pairs for text-video and image-video)
- Loss computation using the JointDiffusionModel
- Optimization & training loop with evaluation metrics
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from onyx.config import Config
from onyx.text_encoder import TextEncoder
from onyx.video_tokenizer import VideoTokenizer
from onyx.joint_model import JointDiffusionModel

# Dummy Dataset for synthetic training data
class SyntheticVideoDataset(Dataset):
    def __init__(self, num_samples=100):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Simulate video frames: [T, C, H, W]
        video = torch.randn(Config.NUM_FRAMES, 3, Config.FRAME_SIZE[1], Config.FRAME_SIZE[0])
        # Simulate text prompt as random integer sequences
        text = torch.randint(0, 10000, (16,))
        # Simulate motion latent as random noise
        motion = torch.randn(Config.NUM_FRAMES, Config.MOTION_TOKEN_DIM)
        return video, text, motion

class Trainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.text_encoder = TextEncoder().to(self.device)
        self.video_tokenizer = VideoTokenizer().to(self.device)
        self.diffusion_model = JointDiffusionModel().to(self.device)

        self.optimizer = optim.Adam(
            list(self.text_encoder.parameters()) +
            list(self.video_tokenizer.parameters()) +
            list(self.diffusion_model.parameters()),
            lr=Config.LEARNING_RATE
        )

        self.dataset = SyntheticVideoDataset()
        self.loader = DataLoader(self.dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    def train(self):
        self.diffusion_model.train()
        for epoch in range(Config.NUM_EPOCHS):
            epoch_loss = 0.0
            for batch in self.loader:
                video, text, motion = batch
                video = video.to(self.device)
                text = text.to(self.device)
                motion = motion.to(self.device)

                # Encode text to get conditioning tokens
                text_encoding = self.text_encoder(text)  # [B, L, embed_dim]
                # Tokenize video frames
                video_tokens = self.video_tokenizer(video)  # [B, T, token_dim]

                # Dummy time noise level (random for training)
                t = torch.rand(video_tokens.size(0), 1).to(self.device)

                # For joint training, split tokens into appearance (x) and motion are provided separately
                # Here, we simply simulate x1, x0 and d1, d0 as video_tokens and motion respectively.
                x1 = video_tokens
                x0 = torch.randn_like(video_tokens)
                d1 = motion
                d0 = torch.randn_like(motion)

                self.optimizer.zero_grad()
                loss = self.diffusion_model.compute_loss(x1, x0, text_encoding, t, d1=d1, d0=d0, joint=True)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}, Loss: {epoch_loss/len(self.loader):.4f}")

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()