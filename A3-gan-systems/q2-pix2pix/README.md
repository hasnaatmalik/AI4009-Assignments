# Question 2: Pix2Pix

This folder contains the notebook for training a U-Net + PatchGAN Pix2Pix architecture, alongside a Gradio app for inference on sketches/grayscale images.

## Files
- `notebook.ipynb`: Kaggle-ready Jupyter Notebook. Computes L1 and Adversarial losses, saves samples in triplets (Sketch | Target | Gen), and outputs final models.
- `app.py`: Gradio app to load `.pth` checkpoints and generate image translations from user sketches. Optimized for Apple Silicon MPS.

## Running the Inference App Locally (Mac)

1. Place the generated `.pth` checkpoint (`pix2pix_generator_100.pth`) inside this folder (`q2-pix2pix/`). 
   (Note: Adjust the epoch number in `app.py` if your checkpoint name varies).
2. Run the Gradio app:
   ```bash
   python3 app.py
   ```
3. Open the provided `localhost` URL in your browser.
4. Upload a sketch and hit "Translate to Photo".

## Models Summary
- **Generator**: U-Net structure with skip connections between encoder blocks and corresponding decoder blocks. Dropout on the first 3 decoder layers.
- **Discriminator**: 30x30 PatchGAN evaluating $N \times N$ overlapping regions. Loss computed across the matrix.
