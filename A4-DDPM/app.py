"""
DDPM Inference App — Apple M2 / MPS + Gradio
=============================================
Run:  python app.py
Then open the URL printed in the terminal.
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gradio as gr
from PIL import Image
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Device — MPS on Apple Silicon, fallback to CPU
# ─────────────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅  Using Apple MPS (Metal Performance Shaders)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅  Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("⚠️   MPS not available — using CPU (will be slow)")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Config  — must match what you used during training on Kaggle
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    IMAGE_SIZE    = 128       # change to 256 if you trained at 256
    T             = 300       # must match training T
    BETA_START    = 1e-4
    BETA_END      = 0.02
    SCHEDULE      = "cosine"  # "linear" or "cosine"
    BASE_CHANNELS = 64
    CHANNEL_MULTS = (1, 2, 4)
    NUM_RES_BLOCKS= 2
    # Path to your downloaded checkpoint from Kaggle
    CKPT_PATH     = "ddpm_ckpt.pt"

cfg = Config()

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Noise Schedule
# ─────────────────────────────────────────────────────────────────────────────
class NoiseSchedule:
    def __init__(self, T, beta_start, beta_end, schedule):
        self.T = T
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, T)
        else:  # cosine
            s     = 0.008
            steps = T + 1
            x     = torch.linspace(0, T, steps)
            ac    = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
            ac    = ac / ac[0]
            betas = torch.clamp(1 - ac[1:] / ac[:-1], 1e-4, 0.9999)

        alphas     = 1.0 - betas
        ac         = torch.cumprod(alphas, 0)
        ac_prev    = F.pad(ac[:-1], (1, 0), value=1.0)

        self.betas                    = betas
        self.alphas_cumprod           = ac
        self.sqrt_alphas_cumprod      = ac.sqrt()
        self.sqrt_one_minus_alphas_cp = (1 - ac).sqrt()
        self.sqrt_recip_alphas_cp     = (1.0 / ac).sqrt()
        self.sqrt_recip_m1_alphas_cp  = (1.0 / ac - 1).sqrt()
        self.posterior_variance       = betas * (1 - ac_prev) / (1 - ac)
        self.posterior_log_var_clipped= torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_c1        = betas * ac_prev.sqrt() / (1 - ac)
        self.posterior_mean_c2        = (1 - ac_prev) * alphas.sqrt() / (1 - ac)

    def _get(self, arr, t, shape):
        out = arr.to(t.device)[t]
        return out.reshape(t.shape[0], *((1,) * (len(shape) - 1)))

    def q_sample(self, x0, t, noise=None):
        noise = noise if noise is not None else torch.randn_like(x0)
        return (
            self._get(self.sqrt_alphas_cumprod, t, x0.shape) * x0
            + self._get(self.sqrt_one_minus_alphas_cp, t, x0.shape) * noise
        ), noise


ns = NoiseSchedule(cfg.T, cfg.BETA_START, cfg.BETA_END, cfg.SCHEDULE)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  U-Net  (identical architecture to training — must match exactly)
# ─────────────────────────────────────────────────────────────────────────────
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim  = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim * 4)
        )

    def forward(self, t):
        half  = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        args  = t[:, None].float() * freqs[None]
        emb   = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.proj(emb)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, dropout=0.1):
        super().__init__()
        self.norm1    = nn.GroupNorm(8, in_ch)
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj= nn.Linear(time_dim, out_ch)
        self.norm2    = nn.GroupNorm(8, out_ch)
        self.drop     = nn.Dropout(dropout)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.res_conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm  = nn.GroupNorm(8, ch)
        self.qkv   = nn.Conv2d(ch, ch * 3, 1)
        self.proj  = nn.Conv2d(ch, ch, 1)
        self.scale = ch ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        h   = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) * self.scale, dim=-1)
        h    = torch.bmm(v, attn.transpose(1, 2)).reshape(B, C, H, W)
        return x + self.proj(h)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x): return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)
    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64,
                 channel_mults=(1, 2, 4), num_res_blocks=2, T=300):
        super().__init__()
        time_dim = base_channels * 4
        self.time_mlp  = SinusoidalTimeEmbedding(base_channels)
        channels       = [base_channels * m for m in channel_mults]
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder
        self.downs, self.downsamps, self.skip_channels = nn.ModuleList(), nn.ModuleList(), []
        in_ch = base_channels
        for i, out_ch in enumerate(channels):
            blks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blks.append(ResidualBlock(in_ch, out_ch, time_dim))
                in_ch = out_ch
            self.downs.append(blks)
            self.skip_channels.append(out_ch)
            self.downsamps.append(Downsample(in_ch) if i != len(channels)-1 else nn.Identity())

        # Bottleneck
        mid = channels[-1]
        self.mid_block1 = ResidualBlock(mid, mid, time_dim)
        self.mid_attn   = AttentionBlock(mid)
        self.mid_block2 = ResidualBlock(mid, mid, time_dim)

        # Decoder
        self.ups, self.upsamps = nn.ModuleList(), nn.ModuleList()
        for i, out_ch in enumerate(reversed(channels)):
            skip_ch = self.skip_channels[-(i+1)]
            blks    = nn.ModuleList()
            for j in range(num_res_blocks):
                res_in = in_ch + (skip_ch if j == 0 else 0)
                blks.append(ResidualBlock(res_in, out_ch, time_dim))
                in_ch = out_ch
            self.ups.append(blks)
            self.upsamps.append(Upsample(in_ch) if i != len(channels)-1 else nn.Identity())

        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_conv = nn.Conv2d(base_channels, in_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x     = self.init_conv(x)
        skips = []
        for blks, ds in zip(self.downs, self.downsamps):
            for blk in blks: x = blk(x, t_emb)
            skips.append(x)
            x = ds(x)
        x = self.mid_block2(self.mid_attn(self.mid_block1(x, t_emb)), t_emb)
        for blks, us, skip in zip(self.ups, self.upsamps, reversed(skips)):
            x = torch.cat([x, skip], dim=1)
            for blk in blks: x = blk(x, t_emb)
            x = us(x)
        return self.out_conv(F.silu(self.out_norm(x)))


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Load checkpoint
# ─────────────────────────────────────────────────────────────────────────────
def load_model():
    model = UNet(
        in_channels   = 3,
        base_channels = cfg.BASE_CHANNELS,
        channel_mults = cfg.CHANNEL_MULTS,
        num_res_blocks= cfg.NUM_RES_BLOCKS,
        T             = cfg.T,
    ).to(device)

    if not os.path.exists(cfg.CKPT_PATH):
        raise FileNotFoundError(
            f"\nCheckpoint not found at '{cfg.CKPT_PATH}'\n"
            "    Download 'ddpm_ckpt.pt' from your Kaggle output and place it\n"
            "    in the same folder as app.py.\n"
        )

    ckpt = torch.load(cfg.CKPT_PATH, map_location="cpu")
    # Handle DataParallel prefix "module." in keys
    state = ckpt.get("model", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    print(f"Checkpoint loaded from '{cfg.CKPT_PATH}'")
    return model


model = load_model()

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Sampling helpers
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def p_sample_step(model, x, t_idx, ns):
    """Single reverse diffusion step."""
    B      = x.shape[0]
    t_b    = torch.full((B,), t_idx, device=device, dtype=torch.long)
    pred   = model(x, t_b)

    def g(arr): return ns._get(arr.to(device), t_b, x.shape)

    x0_pred = (g(ns.sqrt_recip_alphas_cp) * x - g(ns.sqrt_recip_m1_alphas_cp) * pred).clamp(-1, 1)
    mean    = g(ns.posterior_mean_c1) * x0_pred + g(ns.posterior_mean_c2) * x

    if t_idx == 0:
        return mean
    noise = torch.randn_like(x)
    return mean + (0.5 * g(ns.posterior_log_var_clipped)).exp() * noise


def tensor_to_pil(t):
    """Convert (3, H, W) tensor in [-1,1] to PIL Image."""
    arr = ((t.cpu().float() * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Gradio inference functions
# ─────────────────────────────────────────────────────────────────────────────
def generate_from_noise(num_images: int, show_steps: int, start_step: int):
    """
    Generate `num_images` images from pure Gaussian noise.
    Returns: list of final PIL images + list of step PIL images (for first image).
    """
    x = torch.randn(num_images, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, device=device)

    collect_at = set(
        torch.linspace(cfg.T - 1, 0, show_steps).long().tolist()
    )
    step_imgs  = []

    for t in tqdm(reversed(range(cfg.T)), desc="Generating", total=cfg.T, leave=False):
        x = p_sample_step(model, x, t, ns)
        if t in collect_at:
            step_imgs.append(tensor_to_pil(x[0]))  # track first image

    final_imgs = [tensor_to_pil(x[i]) for i in range(num_images)]
    return final_imgs, list(reversed(step_imgs))   # show noise→image order


def reconstruct_image(input_image: Image.Image, noise_level: int, show_steps: int):
    """
    Reconstruct a user-uploaded image via partial noising + denoising.
    """
    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        T.CenterCrop(cfg.IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    x0 = transform(input_image.convert("RGB")).unsqueeze(0).to(device)
    t  = torch.tensor([noise_level], device=device).long()
    xt, _ = ns.q_sample(x0, t)

    x = xt
    collect_at = set(torch.linspace(noise_level - 1, 0, show_steps).long().tolist())
    step_imgs  = []

    for step in tqdm(reversed(range(noise_level)), desc="Reconstructing", total=noise_level, leave=False):
        x = p_sample_step(model, x, step, ns)
        if step in collect_at:
            step_imgs.append(tensor_to_pil(x[0]))

    return tensor_to_pil(x[0]), list(reversed(step_imgs))


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Gradio UI
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
body { font-family: 'SF Pro Display', -apple-system, sans-serif; }
.gr-button-primary { background: #1a1a2e !important; }
h1 { text-align: center; }
.tab-nav button { font-size: 15px; font-weight: 600; }
"""

with gr.Blocks(title="DDPM Image Generator", css=CSS, theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # 🌫️ DDPM — Diffusion Model Inference
        **Denoising Diffusion Probabilistic Model** trained on CelebA-HQ | Running on Apple MPS
        """
    )

    with gr.Tabs():

        # ── Tab 1: Generate from noise ────────────────────────────────────────
        with gr.TabItem("🎲 Generate from Noise"):
            gr.Markdown("Start from pure Gaussian noise and watch the model paint an image.")

            with gr.Row():
                with gr.Column(scale=1):
                    num_imgs    = gr.Slider(1, 5, value=3, step=1,  label="Number of images")
                    show_steps  = gr.Slider(3, 10, value=5, step=1, label="Denoising steps to display")
                    gen_btn     = gr.Button("✨ Generate", variant="primary")

                with gr.Column(scale=2):
                    gen_gallery = gr.Gallery(label="Generated Images", columns=3, height=320)

            gr.Markdown("### 🔬 Denoising Process (first image)")
            step_gallery = gr.Gallery(label="Noise → Image", columns=5, height=200)

            def run_generation(n, steps):
                finals, intermediates = generate_from_noise(n, steps, cfg.T)
                return finals, intermediates

            gen_btn.click(
                fn=run_generation,
                inputs=[num_imgs, show_steps],
                outputs=[gen_gallery, step_gallery],
            )

        # ── Tab 2: Reconstruct uploaded image ─────────────────────────────────
        with gr.TabItem("🖼️ Reconstruct Image"):
            gr.Markdown(
                "Upload any image. The model will noise it partway, then denoise it — "
                "producing a 'diffusion reconstruction'."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    upload      = gr.Image(type="pil",  label="Upload Image")
                    noise_lvl   = gr.Slider(50, cfg.T, value=200, step=10,
                                            label=f"Noise level (0 = clean, {cfg.T} = pure noise)")
                    rec_steps   = gr.Slider(3, 10, value=5, step=1,
                                            label="Steps to display")
                    rec_btn     = gr.Button("🔄 Reconstruct", variant="primary")

                with gr.Column(scale=1):
                    rec_output  = gr.Image(label="Reconstructed Image")

            gr.Markdown("### 🔬 Reconstruction Steps")
            rec_step_gallery = gr.Gallery(label="Noise → Reconstruction", columns=5, height=200)

            def run_reconstruction(img, lvl, steps):
                if img is None:
                    raise gr.Error("Please upload an image first.")
                recon, intermediates = reconstruct_image(img, int(lvl), steps)
                return recon, intermediates

            rec_btn.click(
                fn=run_reconstruction,
                inputs=[upload, noise_lvl, rec_steps],
                outputs=[rec_output, rec_step_gallery],
            )

        # ── Tab 3: Info ────────────────────────────────────────────────────────
        with gr.TabItem("ℹ️ Model Info"):
            total_params = sum(p.numel() for p in model.parameters()) / 1e6
            gr.Markdown(f"""
            ### Model Details
            | Property | Value |
            |---|---|
            | Architecture | U-Net (DDPM) |
            | Image Size | {cfg.IMAGE_SIZE}×{cfg.IMAGE_SIZE} |
            | Timesteps (T) | {cfg.T} |
            | Noise Schedule | {cfg.SCHEDULE.capitalize()} |
            | Channels | 64 → 128 → 256 |
            | Parameters | {total_params:.2f}M |
            | Device | {str(device).upper()} |
            | Checkpoint | `{cfg.CKPT_PATH}` |

            ### What is DDPM?
            A **Denoising Diffusion Probabilistic Model** learns to reverse a gradual noising process.
            During training, noise is added to images step-by-step. The U-Net learns to predict
            and remove that noise — so at inference time, starting from pure Gaussian noise,
            it can generate realistic images by iteratively denoising.
            """)

    gr.Markdown(
        "<center><sub>Built with PyTorch + Gradio | NUCES AI4009 — Spring 2026</sub></center>"
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,          # set True to get a public gradio.live link
        inbrowser=True,       # auto-opens browser tab on Mac
    )