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
        # Extended projection matrices for joint appearance-motion modeling
        self.W_in_plus = nn.Linear(Config.VIDEO_TOKEN_DIM + Config.MOTION_TOKEN_DIM, Config.TRANSFORMER_DIM)
        self.W_out_plus = nn.Linear(Config.TRANSFORMER_DIM, Config.VIDEO_TOKEN_DIM + Config.MOTION_TOKEN_DIM)

    def forward_appearance(self, x_t, y, t):
        """
        Computes the model's prediction as in Equation (4):
        u(x_t, y, t; θ) = M(x_t * W_in, y, t; θ) * W_out
        Here, y is the conditioning (text tokens).
        """
        # project tokens
        x_proj = self.W_in(x_t)
        # For simplicity, we assume y and t are concatenated as extra tokens.
        conditioning = torch.cat([y, t], dim=1)
        # Transformer expects input [S, B, E], so we transpose.
        src = x_proj.transpose(0, 1)
        tgt = conditioning.transpose(0, 1)
        transformer_out = self.transformer(src, tgt)
        transformer_out = transformer_out.transpose(0, 1)
        prediction = self.W_out(transformer_out)
        return prediction

    def forward_joint(self, x_t, d_t, y, t):
        """
        Computes the model joint prediction as in Equation (5):
        u+([x_t, d_t], y, t; θ') = M([x_t, d_t]*W_in_plus, y, t; θ') * W_out_plus
        """
        joint_input = torch.cat([x_t, d_t], dim=-1)
        joint_proj = self.W_in_plus(joint_input)
        conditioning = torch.cat([y, t], dim=1)
        src = joint_proj.transpose(0, 1)
        tgt = conditioning.transpose(0, 1)
        transformer_out = self.transformer(src, tgt)
        transformer_out = transformer_out.transpose(0, 1)
        joint_prediction = self.W_out_plus(transformer_out)
        return joint_prediction

    def compute_loss(self, x1, x0, y, t, d1=None, d0=None, joint=False):
        """
        Compute training loss.
        If joint==False, use Equation (3):
         L = E[ || u(x_t, y, t;θ) - v_t ||^2 ]
        with v_t = x1 - x0.
        If joint==True, use extended Equation (6):
         v_t+ = [v_xt, v_dt] and compare with u+([x_t,d_t], y, t;θ')
        """
        # Sample a noise level uniformly in [0, 1]
        noise_level = t  # t is provided as noise level in [0,1]
        x_t = noise_level * x1 + (1 - noise_level) * x0  # Equation (1)
        v_t = x1 - x0  # Equation (2)

        if not joint:
            prediction = self.forward_appearance(x_t, y, t)
            loss = ((prediction - v_t) ** 2).mean()
        else:
            # d_t = noise applied to motion latent d; compute joint input.
            d_t = noise_level * d1 + (1 - noise_level) * d0
            joint_input = torch.cat([x_t, d_t], dim=-1)
            # Create joint velocity target v_t+ = [x1 - x0, d1 - d0]
            v_t_plus = torch.cat([x1 - x0, d1 - d0], dim=-1)
            prediction = self.forward_joint(x_t, d_t, y, t)
            loss = ((prediction - v_t_plus) ** 2).mean()
        return loss

    def inner_guidance(self, x_t, d_t, y, w1=Config.GUIDANCE_SCALE_TEXT, w2=Config.GUIDANCE_SCALE_MOTION):
        """
        Implements the inner-guidance sampling modification:
        ∇ log p~(x_t, d_t|y)
         = (1+w1+w2)∇ log p(x_t,d_t|y)
           - w1 ∇ log p(x_t,d_t)
           - w2 ∇ log p(x_t|y)
        In practice, this function uses the gradients computed by the model
        and scales them appropriately.
        """
        # For this dummy implementation we assume gradients are produced via autograd.
        # In practice, you would compute (or approximate) the three score functions separately.
        x_t.requires_grad_(True)
        d_t.requires_grad_(True)
        # log p(x_t,d_t|y) prediction
        logp_joint = self.forward_joint(x_t, d_t, y, torch.zeros_like(y))
        score_joint = torch.autograd.grad(logp_joint.sum(), (x_t, d_t), create_graph=True)
        # log p(x_t,d_t) prediction
        logp_uncond = self.forward_joint(x_t, d_t, torch.zeros_like(y), torch.zeros_like(y))
        score_uncond = torch.autograd.grad(logp_uncond.sum(), (x_t, d_t), create_graph=True)
        # log p(x_t|y) prediction (only appearance)
        logp_app = self.forward_appearance(x_t, y, torch.zeros_like(y))
        score_app = torch.autograd.grad(logp_app.sum(), x_t, create_graph=True)
        # Combine scores following the log derivative formula
        guided_score_x = (1 + w1 + w2) * score_joint[0] - w1 * score_uncond[0] - w2 * score_app[0]
        guided_score_d = (1 + w1 + w2) * score_joint[1] - w1 * score_uncond[1]
        return guided_score_x, guided_score_d