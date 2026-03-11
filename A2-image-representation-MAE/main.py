import os
import math
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import gradio as gr

# Setup Device - prioritize MPS for Mac M2
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# --- Model Definitions (from original notebook) ---

class PatchEmbedding(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
    
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class MAE(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3,
                 enc_dim=768, enc_layers=12, enc_heads=12,
                 dec_dim=384, dec_layers=12, dec_heads=6,
                 mask_ratio=0.75):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        
        # Encoder
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, enc_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, enc_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, enc_dim))
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(enc_dim, enc_heads) for _ in range(enc_layers)
        ])
        self.encoder_norm = nn.LayerNorm(enc_dim)
        
        # Decoder
        self.decoder_embed = nn.Linear(enc_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, dec_dim))
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(dec_dim, dec_heads) for _ in range(dec_layers)
        ])
        self.decoder_norm = nn.LayerNorm(dec_dim)
        self.decoder_pred = nn.Linear(dec_dim, self.patch_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.decoder_pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_module_weights)
    
    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def patchify(self, imgs):
        p = self.patch_size
        B, C, H, W = imgs.shape
        h, w = H // p, W // p
        patches = imgs.reshape(B, C, h, p, w, p)
        patches = patches.permute(0, 2, 4, 3, 5, 1)
        patches = patches.reshape(B, h * w, p * p * C)
        return patches
    
    def unpatchify(self, patches):
        p = self.patch_size
        h = w = int(self.num_patches ** 0.5)
        C = 3
        patches = patches.reshape(-1, h, w, p, p, C)
        imgs = patches.permute(0, 5, 1, 3, 2, 4)
        imgs = imgs.reshape(-1, C, h * p, w * p)
        return imgs
    
    def random_masking(self, x, mask_ratio):
        B, N, D = x.shape
        num_keep = int(N * (1 - mask_ratio))
        
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep = ids_shuffle[:, :num_keep]
        x_visible = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        
        mask = torch.ones(B, N, device=x.device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_visible, mask, ids_restore
    
    def encode(self, x, mask_ratio):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        for block in self.encoder_blocks:
            x = block(x)
        x = self.encoder_norm(x)
        
        return x, mask, ids_restore
    
    def decode(self, x, ids_restore):
        x = self.decoder_embed(x)
        num_masked = self.num_patches + 1 - x.shape[1]
        mask_tokens = self.mask_token.repeat(x.shape[0], num_masked, 1)
        x_no_cls = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        
        x_no_cls = torch.gather(
            x_no_cls, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x_no_cls.shape[2])
        )
        
        x = torch.cat([x[:, :1, :], x_no_cls], dim=1)
        x = x + self.decoder_pos_embed
        
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        
        x = self.decoder_pred(x[:, 1:, :])
        return x
    
    def forward(self, imgs, mask_ratio=None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        
        latent, mask, ids_restore = self.encode(imgs, mask_ratio)
        pred = self.decode(latent, ids_restore)
        
        target = self.patchify(imgs)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        
        return loss, pred, mask

# --- Setup & Inference ---

def load_model(checkpoint_path):
    print(f"Loading model from {checkpoint_path}...")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}. Please download it from Kaggle and place it in this directory.")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    model = MAE(
        image_size=config['image_size'],
        patch_size=config['patch_size'],
        enc_dim=config['enc_dim'],
        enc_layers=config['enc_layers'],
        enc_heads=config['enc_heads'],
        dec_dim=config['dec_dim'],
        dec_layers=config['dec_layers'],
        dec_heads=config['dec_heads'],
        mask_ratio=config['mask_ratio']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, config

# Load globally
try:
    model, cfg = load_model('mae_final.pt')
except FileNotFoundError as e:
    print(e)
    model = None
    cfg = None

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(3, 1, 1).to(device)
    std = torch.tensor(std).view(3, 1, 1).to(device)
    return (tensor * std + mean).clamp(0, 1)

def mae_reconstruct(input_image, mask_ratio):
    if model is None:
        return None, None, None, "Error: Model not loaded. Please ensure 'mae_final.pt' is in the directory."
    if input_image is None:
        return None, None, None, "Error: Please upload a valid image before submitting."
    
    transform = transforms.Compose([
        transforms.Resize((cfg['image_size'], cfg['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    img_pil = Image.fromarray(input_image).convert('RGB')
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        loss, pred, mask = model(img_tensor, mask_ratio=mask_ratio)
    
    target = model.patchify(img_tensor)
    mask_exp = mask.unsqueeze(-1).expand_as(target)
    
    # Masked input
    masked_input = target * (1 - mask_exp)
    masked_img = model.unpatchify(masked_input)[0]
    masked_img = denormalize(masked_img).cpu().permute(1, 2, 0).numpy()
    
    # Full reconstruction
    full_recon = target * (1 - mask_exp) + pred.float() * mask_exp
    recon_img = model.unpatchify(full_recon)[0]
    recon_img = denormalize(recon_img).cpu().permute(1, 2, 0).numpy()
    
    # Original
    orig_img = denormalize(img_tensor[0]).cpu().permute(1, 2, 0).numpy()
    
    # Convert to uint8 for Gradio
    orig_img = (orig_img * 255).astype(np.uint8)
    masked_img = (masked_img * 255).astype(np.uint8)
    recon_img = (recon_img * 255).astype(np.uint8)
    
    return orig_img, masked_img, recon_img, f'Reconstruction MSE Loss: {loss.item():.4f}'

# --- Gradio UI ---

if model is not None:
    demo = gr.Interface(
        fn=mae_reconstruct,
        inputs=[
            gr.Image(label='Upload Image'),
            gr.Slider(minimum=0.1, maximum=0.95, value=0.75, step=0.05,
                      label='Mask Ratio (fraction of patches to mask)'),
        ],
        outputs=[
            gr.Image(label='Original (resized to 224x224)'),
            gr.Image(label='Masked Input'),
            gr.Image(label='Reconstruction'),
            gr.Textbox(label='Loss'),
        ],
        title='Masked Autoencoder (MAE) - Inference',
        description='Upload any image and adjust the mask ratio to see how the MAE reconstructs masked patches. '
                    'The model was trained on TinyImageNet and utilizes MPS acceleration if available.',
    )
    
    if __name__ == "__main__":
        demo.launch(server_name="127.0.0.1")
else:
    print("Failed to initialize Gradio app because the model is missing.")
