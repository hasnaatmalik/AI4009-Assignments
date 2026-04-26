# Assignment 4 — DDPM: Denoising Diffusion Probabilistic Models

**Course:** Generative AI (AI4009) | **Semester:** Spring 2026
**Platform:** Kaggle (GPU T4 ×2) → local inference on Apple MPS / CUDA / CPU

---

## Overview

This assignment implements a **Denoising Diffusion Probabilistic Model (DDPM)** for high-resolution face image generation and reconstruction, trained on the **CelebA-HQ** dataset at 128×128 resolution.

The model learns to reverse a gradual noising process: starting from pure Gaussian noise, it iteratively denoises to produce photorealistic face images.

---

## How It Works

### 1. Forward Diffusion (Noising)
A clean image $x_0$ is progressively corrupted over $T = 300$ timesteps using a **cosine noise schedule**:

$$q(x_t \mid x_0) = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

### 2. Reverse Diffusion (Denoising)
A **U-Net** learns to predict the noise $\epsilon_\theta(x_t, t)$ at each timestep. At inference, the model iterates from $x_T \sim \mathcal{N}(0, I)$ back to $x_0$.

### 3. U-Net Architecture
| Component | Details |
|---|---|
| Input / Output | 3-channel RGB, 128×128 |
| Base channels | 64 → 128 → 256 (multipliers: 1×, 2×, 4×) |
| Residual blocks | 2 per resolution level |
| Time conditioning | Sinusoidal embeddings projected via MLP |
| Attention | Self-attention at bottleneck |
| Down / Up sampling | Strided conv / nearest-neighbour + conv |

### 4. Training Setup
| Hyperparameter | Value |
|---|---|
| Dataset | CelebA-HQ (20 000 images, 128×128) |
| Timesteps T | 300 |
| Noise schedule | Cosine (Nichol & Dhariwal 2021) |
| Epochs | 30 |
| Batch size | 16 |
| Optimizer | AdamW, lr = 2e-4 |
| Mixed precision | ✅ (fp16) |
| Hardware | Kaggle dual T4 GPU |

---

## Tasks Implemented

| Task | Description |
|---|---|
| Forward diffusion | Visualize 5 progressive noising steps |
| Unconditional generation | Sample new face images from pure Gaussian noise |
| Image reconstruction | Noise an image to timestep $t_{\text{start}}$, then denoise back |
| Quantitative evaluation | PSNR & SSIM against original images |
| Gradio inference app | Interactive UI for generation + reconstruction |

---

## Gradio App (`app.py`)

The inference app runs locally on Apple MPS (or CUDA / CPU fallback) and exposes two tabs:

**Tab 1 — Generate from Noise**
- Choose how many images to generate (1–5)
- Visualize intermediate denoising steps

**Tab 2 — Reconstruct Image**
- Upload any image
- Choose noise level (how far to corrupt before denoising)
- See the diffusion reconstruction alongside step-by-step intermediates

### Running the App

> **Prerequisite:** Download `ddpm_ckpt.pt` from your Kaggle output and place it in this folder.

```bash
# From the repo root
source .venv/bin/activate
cd A4-DDPM
python app.py
```

Then open the URL printed in the terminal.

---

## File Structure

```
A4-DDPM/
├── AI_ASS04_DDPM.ipynb   # Training notebook (runs on Kaggle)
├── app.py                 # Gradio inference app (runs locally)
├── ddpm_ckpt.pt           # ⚠️  Model checkpoint — NOT tracked by git (see .gitignore)
└── README.md
```

> `ddpm_ckpt.pt` is excluded from version control via `.gitignore` because it exceeds GitHub's file size limits. Download it separately from the Kaggle output.

---

## Results

| Metric | Value |
|---|---|
| PSNR | logged in `ddpm_outputs/metrics.txt` |
| SSIM | logged in `ddpm_outputs/metrics.txt` |

Sample outputs saved by the notebook: `forward_diffusion.png`, `reconstruction.png`, `full_visualization.png`.

---

## References

- Ho et al. (2020) — [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- Nichol & Dhariwal (2021) — [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)
- CelebA-HQ dataset — Liu et al. (2015)
