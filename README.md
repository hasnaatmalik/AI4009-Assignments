# AI4009 - Advanced Intelligence Assignments

This repository contains assignments for the **AI4009** course. Each assignment is organized into its own subdirectory.

## Assignment List

*   **[A1 - Neural Storyteller](./A1-neural-storyteller/README.md)**: Image Captioning using Encoder-Decoder architecture with Bahdanau Attention.

## Setup Instructions

This project uses a shared virtual environment for all assignments to save disk space and simplify setup.

### 1. Create Virtual Environment (One-time Setup)

Run this from the root `AI4009-Assignments` directory:

```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running Assignments

For future work, simply activate the environment before running any assignment code:

```bash
source .venv/bin/activate
cd [Assignment-Folder]
python app.py
```
