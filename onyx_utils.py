"""
Utility functions for Onyx.
"""

import torch
import numpy as np

def tokens_to_video(video_tokens):
    """
    Convert video tokens back to video frames.
    This dummy implementation simply reshapes tokens into image frames.
    In practice, you would implement a learned decoder to reconstruct high-quality frames.
    """
    # video_tokens: [B, T, token_dim]
    B, T, token_dim = video_tokens.shape
    # For dummy purposes, we map each token vector to a 64x64 grayscale frame
    frames = []
    for t in range(T):
        frame = video_tokens[0, t].detach().cpu().numpy()
        # Normalize and reshape to a 64x64 image (flatten if needed)
        img = np.tile(frame.reshape(8, -1), (8, 8))
        img = (img - img.min()) / (img.max() - img.min() + 1e-5) * 255
        frames.append(np.uint8(img))
    # Stack frames into a video list that imageio can write as a video
    return frames
```