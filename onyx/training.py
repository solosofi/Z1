"""
Training script for Onyx.

This file demonstrates a full training pipeline:
- Prepare the dataset by cloning from a GitHub repository.
- Loss computation using the JointDiffusionModel.
- Optimization & training loop with evaluation metrics.
- Automatic saving of model weights after each epoch, including a git commit and push to the official repo.
"""

import os
import subprocess
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from onyx.config import Config
from onyx.text_encoder import TextEncoder
from onyx.video_tokenizer import VideoTokenizer
from onyx.joint_model import JointDiffusionModel

def prepare_dataset():
    """
    Prepare the dataset by cloning it from GitHub. This function does not use Hugging Face's load_dataset.
    """
    dataset_dir = "data/dataset"
    if not os.path.exists(dataset_dir):
        os.makedirs("data", exist_ok=True)
        clone_url = f"https://github.com/{Config.DATASET_REPO}.git"
        print(f"Cloning dataset from {clone_url} into {dataset_dir}...")
        ret = os.system(f"git clone {clone_url} {dataset_dir}")
        if ret != 0:
            print("Error cloning dataset; falling back to synthetic data.")
    else:
        print(f"Dataset directory '{dataset_dir}' already exists.")

class SyntheticVideoDataset(Dataset):
    """
    Dummy Dataset that can later be adapted to load any video or text-to-video data from the cloned dataset.
    """
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        # This is where custom dataset loading from data/dataset would go.

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Simulate video frames: [T, C, H, W]
        video = torch.randn(Config.NUM_FRAMES, 3, Config.FRAME_SIZE[1], Config.FRAME_SIZE[0])
        # Simulate a text prompt as random integer sequences
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

        # Prepare the dataset by cloning from the configured GitHub repository.
        prepare_dataset()
        self.dataset = SyntheticVideoDataset()
        self.loader = DataLoader(self.dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    def save_weights(self, epoch: int):
        os.makedirs("saved_models", exist_ok=True)
        checkpoint = {
            'text_encoder': self.text_encoder.state_dict(),
            'video_tokenizer': self.video_tokenizer.state_dict(),
            'diffusion_model': self.diffusion_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch + 1
        }
        file_path = f"saved_models/model_epoch_{epoch+1}.pth"
        torch.save(checkpoint, file_path)
        print(f"Saved weights locally for epoch {epoch+1} at {file_path}")
        
        # Commit and push the updated weights file to the official GitHub repository.
        try:
            subprocess.check_call(["git", "add", file_path])
            commit_message = f"Update weights for epoch {epoch+1}"
            subprocess.check_call(["git", "commit", "-m", commit_message])
            subprocess.check_call(["git", "push", "origin", Config.WEIGHTS_SAVE_BRANCH])
            print(f"Pushed weights for epoch {epoch+1} to the official repository.")
        except subprocess.CalledProcessError as e:
            print(f"Git command failed: {e}")

    def train(self):
        self.diffusion_model.train()
        for epoch in range(Config.NUM_EPOCHS):
            epoch_loss = 0.0
            for batch in self.loader:
                video, text, motion = batch
                video = video.to(self.device)
                text = text.to(self.device)
                motion = motion.to(self.device)

                # Encode text to get conditioning tokens.
                text_encoding = self.text_encoder(text)
                # Tokenize video frames.
                video_tokens = self.video_tokenizer(video)

                # Dummy time noise level for training.
                t = torch.rand(video_tokens.size(0), 1).to(self.device)

                x1 = video_tokens
                x0 = torch.randn_like(video_tokens)
                d1 = motion
                d0 = torch.randn_like(motion)

                self.optimizer.zero_grad()
                loss = self.diffusion_model.compute_loss(x1, x0, text_encoding, t, d1=d1, d0=d0, joint=True)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(self.loader)
            print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}, Loss: {avg_loss:.4f}")
            self.save_weights(epoch)

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
