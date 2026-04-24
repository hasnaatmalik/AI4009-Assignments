# Question 1: DCGAN and WGAN-GP

This folder contains the training notebook and inference app for training and running DCGAN and WGAN-GP on Anime/Pokemon datasets.

## Files
- `notebook.ipynb`: Kaggle-ready Jupyter Notebook. Computes losses, saves image grids, and saves generator/discriminator checks. Adapted for multi-GPU training with `torch.cuda.amp` scaling.
- `app.py`: Gradio app to load `.pth` checkpoints and generate arbitrary grids. Optimized for Apple Silicon MPS.

## Running the Inference App Locally (Mac)

1. Place the generated `.pth` checkpoints (`dcgan_generator_epoch_50.pth` and `wgangp_generator_epoch_50.pth`) inside this folder (`q1-dcgan-wgangp/`).
2. Run the Gradio app:
   ```bash
   python3 app.py
   ```
3. Open the provided `localhost` URL in your browser. 
4. The models evaluate on Apple's `mps` backend using stripped DataParallel states.

## Model Checkpoints (Not included)
Since the model `.pth` files are >100MB, they must be trained on Kaggle and downloaded. Place them here before running `app.py`.
