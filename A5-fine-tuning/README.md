# Assignment 5 — Vision Language Model (VLM) Fine-Tuning

**Course:** Generative AI (AI4009) | **Semester:** Spring 2026
**Platform:** Kaggle T4×2

---

## Overview

This assignment focuses on fine-tuning a Vision Language Model (VLM) using QLoRA for a Document-to-Markdown generation task.
The model used is **Qwen2-VL-2B-Instruct**, fine-tuned with 4-bit NF4 quantization to interpret document images and output accurately formatted markdown.

---

## Technical Details

| Component | Details |
|---|---|
| **Model** | Qwen2-VL-2B-Instruct |
| **Fine-tuning** | QLoRA — 4-bit NF4 quantization |
| **LoRA Rank** | 16, alpha=32 |
| **Epochs** | 3 (with early stopping) |
| **Batch / Accum** | 1 / 8 (effective batch = 8) |
| **Learning Rate** | 1e-4 cosine schedule |
| **Dataset** | Nougat Training Dataset Example |
| **Split** | 80% train / 20% val |
| **Outputs** | Loss curves, ROUGE scores, per-sample comparison plots, Gradio app |

---

## Implementation Steps

1. **Dataset Exploration & Preparation**: Loading the Nougat Training Dataset and formatting it into ChatML format suitable for instruct models.
2. **QLoRA Fine-Tuning**: Fine-tuning the 2 billion parameter VLM using Low-Rank Adaptation (LoRA) and 4-bit quantization to fit within Kaggle's dual T4 GPU memory constraints.
3. **Evaluation**: Generating markdown from validation set images and testing on completely unseen images.
4. **Comparison**: Providing a side-by-side analysis of Zero-Shot performance versus the Fine-Tuned model outputs.
5. **Gradio Deployment**: Developing a Gradio interface within the notebook for interactive Document-to-Markdown inference.

---

## File Structure

```
A5-fine-tuning/
├── main.ipynb        # Training, evaluation, and inference notebook
└── README.md
```
