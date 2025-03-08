"""
Joint Diffusion Model for Appearance and Motion.

This model implements:
- Diffusion-based video generation (Equations 1-4)
- Joint appearance-motion modeling (Equations 5-6)
- Inner-guidance mechanism (Equations 7-8)

All components are built from scratch using transformer modules.
"""

import torch
import torch.nn as nn
from onyx.config import Config

def cosine_beta_schedule(t, beta_start, beta_end):
    """
    Simple linear schedule for the beta parameters.
    """
    return beta_start + t * (beta_end - beta_start)

class JointDiffusionModel(nn.Module):
    def __init__(self):
        super(JointDiffusionModel, self).__init__()

        self.transformer = nn.Transformer(
            d_model=Config.TRANSFORMER_DIM,
            nhead=Config.NUM_HEADS,
            num_encoder_layers=Config.NUM_TRANSFORMER_LAYERS,
            num_decoder_layers=Config.NUM_TRANSFORMER_LAYERS,
            dim_feedforward=Config.FFN_DIM
        )
        # Projection matrices for appearance (W_in and W_out)
        self.W_in = nn.Linear(Config.VIDEO_TOKEN_DIM, Config.TRANSFORMER_DIM)
        self.W_out = nn.Linear(Config.TRANSFORMER_DIM, Config.VIDEO_TOKEN_DIM)
        # Extended projection matrices for joint appearance-motion modeling (input dimension = 256+128=384)
        self.W_in_plus = nn.Linear(Config.VIDEO_TOKEN_DIM + Config.MOTION_TOKEN_DIM, Config.TRANSFORMER_DIM)
        self.W_out_plus = nn.Linear(Config.TRANSFORMER_DIM, Config.VIDEO_TOKEN_DIM + Config.MOTION_TOKEN_DIM)

        # Projection layers to map conditioning tokens and time to the transformer dimension.
        # Maps text embeddings (dimension Config.TEXT_EMBED_DIM) to transformer dimension.
        self.y_proj = nn.Linear(Config.TEXT_EMBED_DIM, Config.TRANSFORMER_DIM)
        # Maps scalar time input from shape [B, 1] to [B, d_model].
        self.t_proj = nn.Linear(1, Config.TRANSFORMER_DIM)

    def forward_appearance(self, x_t, y, t):
        """
        Computes the model's prediction as in Equation (4):
        u(x_t, y, t; θ) = M(x_t * W_in, y, t; θ) * W_out

        x_t: [B, T, VIDEO_TOKEN_DIM]
        y: [B, L, TEXT_EMBED_DIM] (conditioning text tokens)
        t: [B, 1] (noise level)

        Projects the video tokens, conditioning text, and time scalar to the transformer dimension.
        """
        # Project video tokens to transformer dimension.
        x_proj = self.W_in(x_t)  # [B, T, d_model]
        # Project conditioning text tokens.
        y_proj = self.y_proj(y)  # [B, L, d_model]
        # Project time and unsqueeze so shape becomes [B, 1, d_model]
        t_proj = self.t_proj(t).unsqueeze(1)
        # Concatenate the projected text tokens and time token along the sequence dimension.
        conditioning = torch.cat([y_proj, t_proj], dim=1)  # [B, L+1, d_model]
        # Transformer expects input shape: [sequence, batch, d_model]
        src = x_proj.transpose(0, 1)  # [T, B, d_model]
        tgt = conditioning.transpose(0, 1)  # [L+1, B, d_model]
        transformer_out = self.transformer(src, tgt)  # [T, B, d_model]
        transformer_out = transformer_out.transpose(0, 1)  # [B, T, d_model]
        prediction = self.W_out(transformer_out)  # [B, T, VIDEO_TOKEN_DIM]
        return prediction

    def forward_joint(self, x_t, d_t, y, t):
        """
        Computes the joint model prediction as in Equation (5):
        u+([x_t, d_t], y, t; θ') = M([x_t, d_t] * W_in_plus, y, t; θ') * W_out_plus

        x_t: [B, T, VIDEO_TOKEN_DIM]
        d_t: [B, T, MOTION_TOKEN_DIM]
        y: [B, L, TEXT_EMBED_DIM]
        t: [B, 1]
        """
        # Concatenate appearance and motion along feature dimension.
        joint_input = torch.cat([x_t, d_t], dim=-1)  # [B, T, VIDEO_TOKEN_DIM + MOTION_TOKEN_DIM]
        # Project joint input to transformer dimension.
        joint_proj = self.W_in_plus(joint_input)  # [B, T, d_model]
        # Project conditioning text tokens.
        y_proj = self.y_proj(y)  # [B, L, d_model]
        # Project time and unsqueeze so shape becomes [B, 1, d_model]
        t_proj = self.t_proj(t).unsqueeze(1)
        # Concatenate conditioning text and time.
        conditioning = torch.cat([y_proj, t_proj], dim=1)  # [B, L+1, d_model]
        src = joint_proj.transpose(0, 1)  # [T, B, d_model]
        tgt = conditioning.transpose(0, 1)  # [L+1, B, d_model]
        transformer_out = self.transformer(src, tgt)  # [T, B, d_model]
        transformer_out = transformer_out.transpose(0, 1)  # [B, T, d_model]
        joint_prediction = self.W_out_plus(transformer_out)  # [B, T, VIDEO_TOKEN_DIM + MOTION_TOKEN_DIM]
        return joint_prediction

    def compute_loss(self, x1, x0, y, t, d1=None, d0=None, joint=False):
        """
        Compute training loss.
        If joint==False, use Equation (3):
          L = E[ || u(x_t, y, t; θ) - (x1 - x0) ||^2 ]
        with v_t = x1 - x0, where x_t = t * x1 + (1 - t) * x0.
        If joint==True, compute d_t similarly and set target as
          v_t+ = [ (x1 - x0), (d1 - d0) ].
        """
        # Reshape noise level t from [B, 1] to [B, 1, 1] for proper broadcasting.
        noise_level = t.view(t.shape[0], 1, 1)
        # Equation (1): x_t = t * x1 + (1 - t) * x0
        x_t = noise_level * x1 + (1 - noise_level) * x0
        # Equation (2): v_t = x1 - x0
        v_t = x1 - x0

        if not joint:
            prediction = self.forward_appearance(x_t, y, t)
            loss = ((prediction - v_t) ** 2).mean()
        else:
            # Compute noise for motion tokens: d_t = t * d1 + (1 - t) * d0
            d_t = noise_level * d1 + (1 - noise_level) * d0
            # Create joint target: v_t+ = [ (x1 - x0), (d1 - d0) ]
            v_t_plus = torch.cat([x1 - x0, d1 - d0], dim=-1)
            prediction = self.forward_joint(x_t, d_t, y, t)
            # Ensure the prediction sequence length matches the target length (video frame tokens)
            prediction = prediction[:, :x1.size(1), :]
            loss = ((prediction - v_t_plus) ** 2).mean()
        return loss

    def inner_guidance(self, x_t, d_t, y, w1=Config.GUIDANCE_SCALE_TEXT, w2=Config.GUIDANCE_SCALE_MOTION):
        """
        Implements the inner-guidance sampling modification:
          ∇log p̃(x_t, d_t | y) =
             (1 + w1 + w2) ∇log p(x_t, d_t|y)
           - w1 ∇log p(x_t, d_t)
           - w2 ∇log p(x_t | y)

        This dummy implementation uses autograd to obtain gradients.
        """
        x_t.requires_grad_(True)
        d_t.requires_grad_(True)
        # Use zeros for unconditional inputs.
        zero_t = torch.zeros_like(t)
        logp_joint = self.forward_joint(x_t, d_t, y, zero_t)
        score_joint = torch.autograd.grad(logp_joint.sum(), (x_t, d_t), create_graph=True)
        logp_uncond = self.forward_joint(x_t, d_t, torch.zeros_like(y), zero_t)
        score_uncond = torch.autograd.grad(logp_uncond.sum(), (x_t, d_t), create_graph=True)
        logp_app = self.forward_appearance(x_t, y, zero_t)
        score_app = torch.autograd.grad(logp_app.sum(), x_t, create_graph=True)
        guided_score_x = (1 + w1 + w2) * score_joint[0] - w1 * score_uncond[0] - w2 * score_app[0]
        guided_score_d = (1 + w1 + w2) * score_joint[1] - w1 * score_uncond[1]
        return guided_score_x, guided_score_d
