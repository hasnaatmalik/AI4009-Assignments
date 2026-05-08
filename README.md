# AI4009 - Generative AI

This repository contains assignments for the **AI4009** course. Each assignment is organized into its own subdirectory.

## Assignment List

*   **[A1 - Neural Storyteller](./A1-neural-storyteller/README.md)**: Image Captioning using Encoder-Decoder architecture with Bahdanau Attention.
*   **[A2 - Image Representation](./A2-image-representation-MAE/README.md)**: Self-Supervised Vision Transformer (Masked Autoencoder / MAE) with Interactive Gradio Demo.
*   **[A3 - GAN Systems](./A3-gan-systems/README.md)**: Implementations of DCGAN, WGAN-GP, Pix2Pix, and CycleGAN with Mac MPS-optimized Gradio inference apps.
*   **[A4 - DDPM](./A4-DDPM/README.md)**: Denoising Diffusion Probabilistic Model trained on CelebA-HQ for unconditional face generation and image reconstruction, with PSNR/SSIM evaluation and a Gradio inference app.
*   **[A5 - VLM Fine-Tuning](./A5-fine-tuning/README.md)**: Vision Language Model (Qwen2-VL-2B-Instruct) fine-tuning with QLoRA for Document-to-Markdown generation.

## Setup Instructions

This project uses a shared virtual environment for all assignments to save disk space and simplify setup.

### 1. Create Virtual Environment (One-time Setup)

Run this from the root `AI4009-Assignments` directory. This environment will be shared across all assignments!

```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Install all dependencies for all projects
pip install -r requirements.txt
```

### 2. Running Assignments

For future work, simply activate the environment before running any assignment code:

```bash
source .venv/bin/activate
cd [Assignment-Folder]
python app.py
```
