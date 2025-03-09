"""
Configuration settings for Onyx.
"""

class Config:
    # Model dimensions & parameters
    TEXT_EMBED_DIM = 512
    VIDEO_TOKEN_DIM = 256
    MOTION_TOKEN_DIM = 128
    NUM_TRANSFORMER_LAYERS = 8
    NUM_HEADS = 8
    FFN_DIM = 1024

    # Diffusion parameters
    NUM_DIFFUSION_STEPS = 1000
    BETA_START = 1e-4
    BETA_END = 0.02

    # Guidance scales (for inner-guidance during sampling)
    GUIDANCE_SCALE_TEXT = 1.0
    GUIDANCE_SCALE_MOTION = 1.0

    # Training parameters
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50

    # Data settings
    FRAME_SIZE = (64, 64)  # width, height
    NUM_FRAMES = 16       # number of frames per video clip

    # Transformer and diffusion projection dimensions
    TRANSFORMER_DIM = VIDEO_TOKEN_DIM  # using video token dim as main transformer dimension

    # Dataset settings
    DATASET = "saiyan-world/Goku-MovieGenBench"
    # Weights save branch for git operations
    WEIGHTS_SAVE_BRANCH = "main"
