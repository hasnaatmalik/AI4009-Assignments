import os
import torch
import torch.nn as nn
import gradio as gr
import torchvision.utils as vutils
from PIL import Image
import numpy as np

# Enable MPS fallback just in case
torch.backends.mps.enable_fallback_kernels = True

device = torch.device("mps" if torch.backends.mps.is_available() 
                      else "cuda" if torch.cuda.is_available() 
                      else "cpu")

z_dim = 100

class DCGAN_Generator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

def load_checkpoint(model, path, device):
    if not os.path.exists(path):
        return False
    state_dict = torch.load(path, map_location=device)
    from collections import OrderedDict
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        new_sd[k.replace("module.", "")] = v
    model.load_state_dict(new_sd)
    model.eval()
    return True

# Initialize models
dcgan_netG = DCGAN_Generator(z_dim).to(device)
wgangp_netG = DCGAN_Generator(z_dim).to(device) # WGAN uses exact same generator structure

dcgan_loaded = load_checkpoint(dcgan_netG, "dcgan_generator_epoch_50.pth", device)
wgangp_loaded = load_checkpoint(wgangp_netG, "wgangp_generator_epoch_50.pth", device)

def generate_images(model_type, num_images, seed):
    if seed >= 0:
        torch.manual_seed(int(seed))
        
    noise = torch.randn(int(num_images), z_dim, 1, 1, device=device)
    
    model = dcgan_netG if model_type == 'DCGAN' else wgangp_netG
    loaded = dcgan_loaded if model_type == 'DCGAN' else wgangp_loaded
    
    if not loaded:
        # Create a dummy image with a warning
        img = Image.new('RGB', (256, 256), color=(255, 0, 0))
        return img, "Checkpoint not found! Please place 'dcgan_generator_epoch_50.pth' or 'wgangp_generator_epoch_50.pth' in this directory."
        
    with torch.no_grad():
        fake = model(noise).detach().cpu()
        
    # Create grid
    grid = vutils.make_grid(fake, padding=2, normalize=True, nrow=min(4, int(num_images)))
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    img = Image.fromarray(ndarr)
    return img, f"Generated {int(num_images)} images successfully."

with gr.Blocks(title="AI4009 A3 Q1: GAN Generators") as demo:
    gr.Markdown("# Q1: DCGAN and WGAN-GP Generator Demo")
    gr.Markdown(f"Running on device: **{device}**")
    
    with gr.Tabs():
        with gr.Tab("DCGAN"):
            gr.Markdown("### DCGAN Inference")
            with gr.Row():
                with gr.Column():
                    dcgan_num = gr.Slider(minimum=1, maximum=16, step=1, value=8, label="Number of Images")
                    dcgan_seed = gr.Number(value=-1, label="Seed (-1 for random)", precision=0)
                    dcgan_btn = gr.Button("Generate DCGAN Images")
                with gr.Column():
                    dcgan_out_img = gr.Image(label="Generated Output")
                    dcgan_out_txt = gr.Textbox(label="Status")
                    
            dcgan_btn.click(fn=lambda n, s: generate_images('DCGAN', n, s), 
                            inputs=[dcgan_num, dcgan_seed], 
                            outputs=[dcgan_out_img, dcgan_out_txt])
                            
        with gr.Tab("WGAN-GP"):
            gr.Markdown("### WGAN-GP Inference")
            with gr.Row():
                with gr.Column():
                    wgan_num = gr.Slider(minimum=1, maximum=16, step=1, value=8, label="Number of Images")
                    wgan_seed = gr.Number(value=-1, label="Seed (-1 for random)", precision=0)
                    wgan_btn = gr.Button("Generate WGAN-GP Images")
                with gr.Column():
                    wgan_out_img = gr.Image(label="Generated Output")
                    wgan_out_txt = gr.Textbox(label="Status")
                    
            wgan_btn.click(fn=lambda n, s: generate_images('WGAN-GP', n, s), 
                           inputs=[wgan_num, wgan_seed], 
                           outputs=[wgan_out_img, wgan_out_txt])

if __name__ == "__main__":
    # Test a simple forward pass implicitly sets context
    print(f"Server starting on {device}...")
    demo.launch()
