"""
Text encoder module.

This encoder converts input text prompts into conditioning tokens.
"""

import torch
import torch.nn as nn
from onyx.config import Config

class TextEncoder(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=Config.TEXT_EMBED_DIM, max_length=128):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_length, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, dim_feedforward=2048)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
    
    def forward(self, text_indices):
        # text_indices: [B, L]
        x = self.embedding(text_indices)  # [B, L, embed_dim]
        x = x + self.positional_encoding[:, :x.size(1), :]
        # Change shape for transformer: [L, B, embed_dim]
        x = x.permute(1, 0, 2)
        encoded = self.transformer(x)  # [L, B, embed_dim]
        # Permute back to [B, L, embed_dim]
        return encoded.permute(1, 0, 2)