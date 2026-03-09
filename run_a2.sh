#!/bin/bash
# Description: This script downloads the required mae_final.pt weights from a public URL or reminds the user how to do it.

# The user is expected to download the weights from Kaggle directly:
# URL: https://www.kaggle.com/datasets/hasnaatmalik/mae-final-weights  (or equivalent)
# For the sake of automation, if we have a direct link we would use `curl` or `wget`.
# Since Kaggle requires authentication for dataset downloads, we will print instructions to the terminal 
# and try to check if the file exists.

FILE="A2-image-representation-MAE/mae_final.pt"

if [ -f "$FILE" ]; then
    echo "✅ Success: $FILE exists."
    echo "Starting Gradio application..."
    cd A2-image-representation-MAE
    python main.py
else
    echo "❌ Error: Model checkpoint not found."
    echo ""
    echo "Please download 'mae_final.pt' from your Kaggle kernel output and save it"
    echo "inside the A2-image-representation-MAE folder."
    echo ""
    echo "Once the file is placed, simply run this script again or run:"
    echo "  cd A2-image-representation-MAE"
    echo "  python main.py"
fi
