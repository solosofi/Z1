"""
Onyx: Unified Video Generation System

This package contains modules for text-to-video synthesis, joint appearance-motion modeling,
diffusion generation, and post-processing.
"""

from .config import Config
from .text_encoder import TextEncoder
from .video_tokenizer import VideoTokenizer
from .joint_model import JointDiffusionModel
from .training import Trainer
from .inference import InferenceEngine
from .evaluation import evaluate_video