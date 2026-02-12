# Assignment 1: Neural Storyteller (Image Captioning)

An AI-powered image captioning system that generates descriptive sentences for identifying images using deep learning.

## Overview

This project implements an **Encoder-Decoder architecture** with **Bahdanau Attention**:
-   **Encoder**: ResNet50 (modified to extract spatial features).
-   **Decoder**: LSTM with Attention mechanism.
-   **Dataset**: Flickr30k.

## Project Structure

-   `AI_ASS01_22F_3389,22F-3354.ipynb`: Jupyter Notebook for training the model on Kaggle.
-   `app.py`: Standalone Gradio web application for inference.
-   `best_model.pth`: Trained model weights (generated from Kaggle).
-   `vocab.pkl`: Vocabulary dictionary (generated from Kaggle).

## How to Run

### 1. Prerequisites (Model Files)
Since training takes hours, you need the trained model files.
1.  Upload the notebook (`.ipynb`) to Kaggle.
2.  Add the **Flickr30k** dataset.
3.  Run the notebook (or at least the first few cells to get `vocab.pkl`).
4.  Download `best_model.pth` and `vocab.pkl` from the Kaggle Output section.
5.  Place them in this directory.

### 2. Run the App
Make sure your virtual environment is activated (see root README).

```bash
# From AI4009-Assignments/A1-neural-storyteller/
python3 app.py
```

This will launch a web interface at `http://127.0.0.1:7860`.

### 3. Features
-   **Greedy Search**: Generates the most likely next word at each step. Fast but can get stuck in loops.
-   **Beam Search**: Keeps track of `k` most likely sequences. Slower but produces better, more coherent sentences.
