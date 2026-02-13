# Neural Storyteller: Image Captioning

This project implements an Image Captioning system using an Attention-based LSTM model with a Streamlit interface.

## Local Setup

1.  **Clone the repository:**
    ```bash
    git clone <https://github.com/hasnaatmalik/AI4009-Assignments/tree/main/A1-neural-storyteller>
    cd <A1-neural-storyteller>
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app:**
    ```bash
    streamlit run app.py
    ```
    *Note: On the first run, the app will reconstruct the large model file from chunks (`model_chunk_*`).*

## Deploying to Streamlit Cloud

1.  **Push to GitHub:**
    Ensure your code is pushed to a GitHub repository. The model file is split into chunks to comply with GitHub's file size limits.

2.  **Sign in to Streamlit Cloud:**
    Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with your GitHub account.

3.  **Deploy:**
    - Click "New app".
    - Select your repository, branch (usually `main`), and the main file path (`app.py`).
    - Click "Deploy".

4.  **Enjoy!**
    Your app should be live in a few minutes.
