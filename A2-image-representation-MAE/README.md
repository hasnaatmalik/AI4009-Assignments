# A2 - Image Representation

This directory contains the code for self-supervised representation learning using a Masked Autoencoder (MAE). The model uses a Vision Transformer (ViT) architecture to randomly mask image patches and reconstruct the missing pixels.

## Requirements

The core assignment logic is developed in `main.ipynb`. We have extracted the inference application into `main.py` which runs a Gradio interface.

### Running Inference Locally

We have provided a helper script in the root directory to make running this app easy!

1.  **Download the Pre-trained Weights:**
    *   Go to your Kaggle Kernel output for this assignment.
    *   Download the locally saved `mae_final.pt` weights file.
    *   Move the `mae_final.pt` file into this directory (`AI4009-Assignments/A2-image-representation-MAE/mae_final.pt`).

2.  **Ensure Dependencies are Installed:**
    Open your terminal in the root `AI4009-Assignments` folder and run:
    ```bash
    source .venv/bin/activate
    pip install -r requirements.txt 
    ```

3.  **Run the Gradio App:**
    You can run the app using the provided helper script from the root folder:
    ```bash
    ./run_a2.sh
    ```
    
    *Alternatively, you can run it directly from this folder:*
    ```bash
    cd A2-image-representation-MAE
    python main.py
    ```

    The Gradio application will automatically utilize the M2 Mac GPU (`mps` backend) and launch in your browser at `http://127.0.0.1:7860`.
