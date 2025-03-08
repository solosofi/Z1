"""
Video tokenizer module.

This module compresses a video (sequence of frames) into a compact set of discrete tokens.
In this update the spatial dimension is aggregated via average pooling so that each frame
is represented by a single token of dimension Config.VIDEO_TOKEN_DIM.
"""

import torch
import torch.nn as nn
from onyx.config import Config

class VideoTokenizer(nn.Module):
    def __init__(self, in_channels=3, token_dim=Config.VIDEO_TOKEN_DIM):
        super(VideoTokenizer, self).__init__()
        # A simple CNN backbone to encode frames into latent tokens.
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
        # Pool spatial dimensions (average over H/4 x W/4)
        tokens = tokens.mean(dim=[2, 3])  # [B*T, token_dim]
        # Reshape back to [B, T, token_dim]
        tokens = tokens.view(B, T, -1)
        return tokens
