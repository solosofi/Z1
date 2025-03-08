"""
Video tokenizer module.

This module compresses a video (sequence of frames) into a compact set of discrete tokens via a causal model.
"""

import torch
import torch.nn as nn
from onyx.config import Config

class VideoTokenizer(nn.Module):
    def __init__(self, in_channels=3, token_dim=Config.VIDEO_TOKEN_DIM):
        super(VideoTokenizer, self).__init__()
        # A simple CNN backbone to encode frames into latent tokens
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # [B, 32, H/2, W/2]
            nn.ReLU(),
            nn.Conv2d(32, token_dim, kernel_size=3, stride=2, padding=1),    # [B, token_dim, H/4, W/4]
            nn.ReLU()
        )
    
    def forward(self, video_frames):
        # video_frames: [B, T, C, H, W]
        B, T, C, H, W = video_frames.shape
        video_frames = video_frames.view(B * T, C, H, W)
        tokens = self.encoder(video_frames)  # [B*T, token_dim, H/4, W/4]
        # Flatten spatial dimensions
        tokens = tokens.view(B, T, tokens.shape[1], -1)  # [B, T, token_dim, spatial]
        # Permute to [B, T, spatial, token_dim]
        tokens = tokens.permute(0, 1, 3, 2)
        # Collapse spatial tokens
        tokens = tokens.flatten(2)  # [B, T, total_tokens]
        return tokens