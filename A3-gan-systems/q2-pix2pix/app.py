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

# Generator Definition
class BlockUNet(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", dropout=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False) if down else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True) if act == "relu" else nn.LeakyReLU(0.2, True)
        )
        self.use_dropout = dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.net(x)
        return self.dropout(x) if self.use_dropout else x
        
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.d1 = nn.Sequential(nn.Conv2d(in_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2))
        self.d2 = BlockUNet(64, 128, down=True, act="leaky")
        self.d3 = BlockUNet(128, 256, down=True, act="leaky")
        self.d4 = BlockUNet(256, 512, down=True, act="leaky")
        self.d5 = BlockUNet(512, 512, down=True, act="leaky")
        self.d6 = BlockUNet(512, 512, down=True, act="leaky")
        self.d7 = BlockUNet(512, 512, down=True, act="leaky")
        self.d8 = nn.Sequential(nn.Conv2d(512, 512, 4, 2, 1), nn.ReLU())

        self.u1 = BlockUNet(512, 512, down=False, act="relu", dropout=True)
        self.u2 = BlockUNet(1024, 512, down=False, act="relu", dropout=True)
        self.u3 = BlockUNet(1024, 512, down=False, act="relu", dropout=True)
        self.u4 = BlockUNet(1024, 512, down=False, act="relu", dropout=False)
        self.u5 = BlockUNet(1024, 256, down=False, act="relu", dropout=False)
        self.u6 = BlockUNet(512, 128, down=False, act="relu", dropout=False)
        self.u7 = BlockUNet(256, 64, down=False, act="relu", dropout=False)
        self.u8 = nn.Sequential(nn.ConvTranspose2d(128, out_channels, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        d8 = self.d8(d7)

        u1 = self.u1(d8)
        u2 = self.u2(torch.cat([u1, d7], 1))
        u3 = self.u3(torch.cat([u2, d6], 1))
        u4 = self.u4(torch.cat([u3, d5], 1))
        u5 = self.u5(torch.cat([u4, d4], 1))
        u6 = self.u6(torch.cat([u5, d3], 1))
        u7 = self.u7(torch.cat([u6, d2], 1))
        u8 = self.u8(torch.cat([u7, d1], 1))
        
        return u8

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

netG = UNetGenerator().to(device)
loaded = load_checkpoint(netG, "pix2pix_generator_100.pth", device) # Adjust epoch accordingly

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def evaluate(input_img):
    if not loaded:
        # Dummy red image as error standard
        img = Image.new('RGB', (256, 256), color=(255, 0, 0))
        return img, "Checkpoint not found! Please place 'pix2pix_generator_100.pth' in this directory."
    
    if input_img is None:
        return None, "Please upload an image"
    
    # Preprocess
    img = input_img.convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        fake_t = netG(img_t).squeeze(0).cpu()
    
    # Unnormalize and to PIL
    fake_t = fake_t * 0.5 + 0.5
    ndarr = fake_t.clamp(0, 1).mul(255).permute(1, 2, 0).byte().numpy()
    out_img = Image.fromarray(ndarr)
    
    return out_img, "Success!"

with gr.Blocks(title="AI4009 A3 Q2: Pix2Pix") as demo:
    gr.Markdown("# Q2: Pix2Pix Sketch Colorization/Translation")
    gr.Markdown(f"Running on device: **{device}**")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Input Sketch/Grayscale Image")
            btn = gr.Button("Translate to Photo")
        with gr.Column():
            output_img = gr.Image(label="Generated Output")
            status = gr.Textbox(label="Status")
            
    btn.click(fn=evaluate, inputs=input_img, outputs=[output_img, status])

if __name__ == "__main__":
    print(f"Server starting on {device}...")
    demo.launch()
