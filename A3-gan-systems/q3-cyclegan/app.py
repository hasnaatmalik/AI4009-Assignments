import os
import torch
import torch.nn as nn
import gradio as gr
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

# Enable MPS fallback just in case
torch.backends.mps.enable_fallback_kernels = True

device = torch.device("mps" if torch.backends.mps.is_available() 
                      else "cuda" if torch.cuda.is_available() 
                      else "cpu")

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3), 
            nn.InstanceNorm2d(dim), 
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3), 
            nn.InstanceNorm2d(dim)
        )
    def forward(self, x): 
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64, num_residuals=6):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, features, 7),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling
        in_features = features
        out_features = in_features * 2
        for _ in range(2):
            layers += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2
            
        # ResBlocks
        for _ in range(num_residuals):
            layers += [ResBlock(in_features)]
            
        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2
            
        # Output layer
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(features, 3, 7),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

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

G_AB = Generator(num_residuals=6).to(device)
G_BA = Generator(num_residuals=6).to(device)

# Load arbitrary epoch checkpoints 
loaded_AB = load_checkpoint(G_AB, "G_AB_100.pth", device) # A -> B (Sketch -> Photo)
loaded_BA = load_checkpoint(G_BA, "G_BA_100.pth", device) # B -> A (Photo -> Sketch)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def evaluate(input_img, direction):
    if not loaded_AB or not loaded_BA:
        img = Image.new('RGB', (128, 128), color=(255, 0, 0))
        return img, "Checkpoints not found! Require G_AB_100.pth and G_BA_100.pth."
        
    if input_img is None:
        return None, "Please upload an image."
        
    img = input_img.convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        if direction == "Sketch -> Photo":
            out_t = G_AB(img_t).squeeze(0).cpu()
        else:
            out_t = G_BA(img_t).squeeze(0).cpu()
            
    out_t = out_t * 0.5 + 0.5
    ndarr = out_t.clamp(0, 1).mul(255).permute(1, 2, 0).byte().numpy()
    res_img = Image.fromarray(ndarr)
    
    return res_img, f"Success! Transformed via {direction}."

with gr.Blocks(title="AI4009 A3 Q3: CycleGAN") as demo:
    gr.Markdown("# Q3: CycleGAN Bidirectional Translation")
    gr.Markdown(f"Running on device: **{device}**")
    
    with gr.Row():
        with gr.Column():
            direction = gr.Radio(choices=["Sketch -> Photo", "Photo -> Sketch"], value="Sketch -> Photo", label="Translation Direction")
            input_img = gr.Image(type="pil", label="Input Image")
            btn = gr.Button("Translate")
        with gr.Column():
            output_img = gr.Image(label="Generated Output")
            status = gr.Textbox(label="Status")
            
    btn.click(fn=evaluate, inputs=[input_img, direction], outputs=[output_img, status])

if __name__ == "__main__":
    print(f"Server starting on {device}...")
    demo.launch()
