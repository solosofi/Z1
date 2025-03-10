"""
Training script for Onyx.

This file demonstrates a training pipeline:
- Loading a video or text-to-video dataset using Hugging Face's datasets library.
- Loss computation using the JointDiffusionModel.
- Optimization & training loop with evaluation metrics.
- Automatic saving of model weights after each epoch, including a git commit and push to the official repository.
"""

import os
import subprocess
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from onyx.config import Config
from onyx.text_encoder import TextEncoder
from onyx.video_tokenizer import VideoTokenizer
from onyx.joint_model import JointDiffusionModel

class VideoTextDataset(Dataset):
    """
    A wrapper dataset to convert Hugging Face's dataset into a torch Dataset.
    Expected to have at least a 'text' field and, optionally, a 'video' field.
    In case of missing video data, synthetic video data will be generated.
    """
    def __init__(self, hf_dataset, num_frames, frame_size, motion_dim):
        self.hf_dataset = hf_dataset
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.motion_dim = motion_dim

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]

        # Process video field.
        if "video" in sample and sample["video"] is not None:
            video_sample = sample["video"]
            # If video_sample is a dict with "array", use that.
            if isinstance(video_sample, dict) and "array" in video_sample:
                video_sample = video_sample["array"]
            # If video_sample is a list, process each frame.
            if isinstance(video_sample, list):
                try:
                    # Convert each frame, calling asnumpy() if available.
                    frames = []
                    for frame in video_sample:
                        if hasattr(frame, "asnumpy"):
                            frame = frame.asnumpy()
                        frame_np = np.array(frame, dtype=np.float32)
                        frames.append(frame_np)
                    video_np = np.stack(frames)
                except Exception as e:
                    print(f"Error converting video sample frames to numpy array: {e}")
                    video_np = np.random.randn(self.num_frames, 3, self.frame_size[1], self.frame_size[0]).astype(np.float32)
            # Otherwise, if video_sample supports asnumpy(), use it.
            elif hasattr(video_sample, "asnumpy"):
                try:
                    video_np = np.array(video_sample.asnumpy(), dtype=np.float32)
                except Exception as e:
                    print(f"Error converting video sample (asnumpy) to numpy array: {e}")
                    video_np = np.random.randn(self.num_frames, 3, self.frame_size[1], self.frame_size[0]).astype(np.float32)
            else:
                # Try converting directly.
                try:
                    video_np = np.array(video_sample, dtype=np.float32)
                except Exception as e:
                    print(f"Error converting video sample to numpy array: {e}")
                    video_np = np.random.randn(self.num_frames, 3, self.frame_size[1], self.frame_size[0]).astype(np.float32)
            video = torch.from_numpy(video_np)
        else:
            video = torch.randn(self.num_frames, 3, self.frame_size[1], self.frame_size[0])

        # Process text: convert text to a sequence of integer token IDs.
        if "text" in sample and sample["text"]:
            text_str = sample["text"]
            # A simple placeholder: convert characters to their ordinal values, padded/truncated to length 16.
            text_tokens = [ord(c) for c in text_str][:16]
            if len(text_tokens) < 16:
                text_tokens += [0] * (16 - len(text_tokens))
            text = torch.tensor(text_tokens, dtype=torch.long)
        else:
            text = torch.randint(0, 10000, (16,))
        
        # Simulate motion latent as random noise.
        motion = torch.randn(self.num_frames, self.motion_dim)
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

        # Load dataset using Hugging Face's load_dataset.
        print(f"Loading dataset {Config.DATASET} ...")
        hf_dataset = load_dataset(Config.DATASET)
        # Choose the 'train' split if available, otherwise the first available split.
        if "train" in hf_dataset:
            train_split = hf_dataset["train"]
        else:
            train_split = hf_dataset[list(hf_dataset.keys())[0]]
        
        self.dataset = VideoTextDataset(
            train_split,
            num_frames=Config.NUM_FRAMES,
            frame_size=Config.FRAME_SIZE,
            motion_dim=Config.MOTION_TOKEN_DIM
        )
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
                video = video.to(self.device, non_blocking=True)
                text = text.to(self.device, non_blocking=True)
                motion = motion.to(self.device, non_blocking=True)

                # Encode text to obtain conditioning tokens.
                text_encoding = self.text_encoder(text)
                # Tokenize video frames.
                video_tokens = self.video_tokenizer(video)

                # Dummy time noise level for training.
                t = torch.rand(video_tokens.size(0), 1).to(self.device)

                # Simulate separate inputs for joint training.
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
