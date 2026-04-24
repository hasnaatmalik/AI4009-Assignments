# Question 3: CycleGAN

This directory implements Unpaired Image-to-Image translation utilizing the CycleGAN architecture for transforming Sketches into Photos (and optionally vice-versa).

## Files
- `notebook.ipynb`: Kaggle-ready Jupyter Notebook. Sets up Domain A & B mapping, implements LSGAN identity/cycle-consistency losses, and trains 2 Generators and 2 Discriminators alternately. Outputs `.pth` checkpoints.
- `app.py`: Gradio app to perform bidirectional translation (Sketch -> Photo or Photo -> Sketch) based on radio buttons. Optimized for Apple Silicon MPS.

## Running the Inference App Locally (Mac)

1. Place the generated `.pth` checkpoints (`G_AB_100.pth` and `G_BA_100.pth`) inside this folder (`q3-cyclegan/`). 
   (Adjust the epoch number in `app.py` if your checkpoints use a different epoch).
2. Run the Gradio app:
   ```bash
   python3 app.py
   ```
3. Open the provided `localhost` URL in your browser.
4. Select the translation direction, upload an image, and infer!

## Architecture Summary
- **Generators (`G_AB` and `G_BA`)**: ResNet blocks (6 layers) with Instance Normalization for high-fidelity translation without batch dependencies.
- **Discriminators (`D_A` and `D_B`)**: PatchGAN operating on image representations, updated via Mean Squared Error (LSGAN) avoiding standard BCE vanishing gradients.
