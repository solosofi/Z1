```markdown
# Onyx: Unified Video Generation System

Onyx is a unified video generation system built from scratch using PyTorch. It integrates:
- Arbitrary-length text-to-video synthesis (Phenaki-inspired)
- Joint appearance–motion modeling with explicit motion guidance (VideoJAM-inspired)
- Diffusion-based video generation via transformer scaling (Sora-inspired)
- Advanced transformer modules for motion dynamics, retargeting, and spatiotemporal prediction
- Transformer-based modules for video inpainting, style transfer, super-resolution, and post-processing

## Project Structure

```
Onyx/
├── requirements.txt
├── README.md
└── onyx
    ├── __init__.py
    ├── config.py
    ├── text_encoder.py
    ├── video_tokenizer.py
    ├── joint_model.py
    ├── training.py
    ├── inference.py
    ├── evaluation.py
    └── utils.py
```

## Setup

1. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run training from Colab or locally:

   ```bash
   python -m onyx.training
   ```

3. Run inference:

   ```bash
   python -m onyx.inference --prompt "A ballet dancer twirls gracefully in a city plaza at sunset, transitioning into dynamic urban breakdance" --output video.mp4
   ```

## Overview

- **Text-to-Video Generation**: A `TextEncoder` converts text prompts to conditioning tokens.
- **Video Tokenization**: A `VideoTokenizer` compresses video frames into discrete tokens.
- **Joint Appearance–Motion Modeling**: The `JointDiffusionModel` performs diffusion-based denoising for both appearance and motion.
- **Diffusion Process & Guidance**: Implements equations for the diffusion process (Equations 1–4) with extensions for joint appearance–motion (Equations 5–6) and inner-guidance (Equations 7–8).
- **Training & Inference Pipelines**: Complete pipelines are provided.
- **Evaluation**: Evaluation scripts measure appearance quality, motion coherence, text alignment, and physical plausibility.

Detailed inline documentation in each source file provides further implementation details.
```
