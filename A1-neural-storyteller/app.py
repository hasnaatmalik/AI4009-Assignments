import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pickle
import os
import streamlit as st
import re
import collections

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Vocabulary Class (Must match the one in the notebook) ---
class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.stoi = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.split()

    def build_vocabulary(self, sentence_list):
        frequencies = collections.Counter()
        idx = 4
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [self.stoi[token] if token in self.stoi else self.stoi["<unk>"] for token in tokenized_text]


# --- Model Architecture ---

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        self.inception = models.inception_v3(pretrained=True)
        self.inception.aux_logits = False
        
        # Using ResNet50 as per the notebook implementation
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2] # Remove avgpool and fc
        self.resnet = nn.Sequential(*modules)
        
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images) # (batch, 2048, 7, 7)
        features = features.permute(0, 2, 3, 1) # (batch, 7, 7, 2048)
        features = features.view(features.size(0), -1, features.size(3)) # (batch, 49, 2048)
        return features

class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(BahdanauAttention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out) # (batch, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden) # (batch, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))) # (batch, num_pixels, 1)
        alpha = self.softmax(att) # (batch, num_pixels, 1)
        context = (encoder_out * alpha).sum(dim=1) # (batch, encoder_dim)
        return context, alpha

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, encoder_dim=2048, attention_dim=256):
        super(DecoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.attention = BahdanauAttention(encoder_dim, hidden_size, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.init_h = nn.Linear(encoder_dim, hidden_size)
        self.init_c = nn.Linear(encoder_dim, hidden_size)
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim, hidden_size)
        self.f_beta = nn.Linear(hidden_size, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, features, captions):
        embeddings = self.embedding(captions)
        h, c = self.init_hidden_state(features)
        
        seq_len = captions.size(1) - 1
        batch_size = captions.size(0)
        num_features = features.size(1)
        
        preds = torch.zeros(batch_size, seq_len, self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, seq_len, num_features).to(device)
        
        for s in range(seq_len):
            context, alpha = self.attention(features, h)
            gate = self.sigmoid(self.f_beta(h))
            gated_context = gate * context
            
            lstm_input = torch.cat((embeddings[:, s, :], gated_context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            
            output = self.fc(self.dropout(h))
            preds[:, s, :] = output
            alphas[:, s, :] = alpha.squeeze(2)
            
        return preds, alphas

    def generate_caption(self, features, vocab, max_len=20):
        # Inference (Greedy)
        batch_size = features.size(0)
        h, c = self.init_hidden_state(features)
        
        word = torch.tensor(vocab.stoi['<start>']).view(1).to(device)
        embeds = self.embedding(word)
        
        captions = []
        alphas = []
        
        for i in range(max_len):
            context, alpha = self.attention(features, h)
            gate = self.sigmoid(self.f_beta(h))
            gated_context = gate * context
            
            lstm_input = torch.cat((embeds, gated_context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            
            output = self.fc(h)
            predicted_word_idx = output.argmax(dim=1)
            
            captions.append(predicted_word_idx.item())
            alphas.append(alpha)
            
            if vocab.itos[predicted_word_idx.item()] == "<end>":
                break
            
            embeds = self.embedding(predicted_word_idx)
        
        return [vocab.itos[idx] for idx in captions], alphas
    
    def generate_caption_beam_search(self, features, vocab, beam_width=5, max_len=20):
        # Simple Beam Search Implementation
        k = beam_width
        h, c = self.init_hidden_state(features)
        
        start_token = vocab.stoi['<start>']
        sequences = [[0.0, start_token, h, c, [start_token]]]
        
        completed_sequences = []
        
        for step in range(max_len):
            all_candidates = []
            
            for score, word_idx, h_prev, c_prev, seq in sequences:
                if word_idx == vocab.stoi['<end>']:
                    completed_sequences.append((score, seq))
                    continue
                
                word_tensor = torch.tensor([word_idx]).to(device)
                embeds = self.embedding(word_tensor)
                
                # Attention step
                context, _ = self.attention(features, h_prev)
                gate = self.sigmoid(self.f_beta(h_prev))
                gated_context = gate * context
                
                lstm_input = torch.cat((embeds, gated_context), dim=1)
                h_new, c_new = self.lstm_cell(lstm_input, (h_prev, c_prev))
                
                output = self.fc(h_new)
                log_probs = F.log_softmax(output, dim=1)
                
                # Get top k words
                topk_probs, topk_indices = log_probs.topk(k, dim=1)
                
                for i in range(k):
                    word = topk_indices[0][i].item()
                    prob = topk_probs[0][i].item()
                    all_candidates.append((score + prob, word, h_new, c_new, seq + [word]))
            
            if not all_candidates:
                break
                
            # Select top k
            sequences = sorted(all_candidates, key=lambda x: x[0], reverse=True)[:k]
            
            # If all sequences end with <end>, break
            if all(seq[1] == vocab.stoi['<end>'] for seq in sequences):
                completed_sequences.extend([(s[0], s[4]) for s in sequences])
                break
        
        if not completed_sequences:
             completed_sequences = [(s[0], s[4]) for s in sequences]

        # Get best sequence
        best_seq = sorted(completed_sequences, key=lambda x: x[0], reverse=True)[0][1]
        
        return [vocab.itos[idx] for idx in best_seq[1:]] # Skip <start>


# --- Helper Functions ---

class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, attention_dim):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, attention_dim=attention_dim)
        
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs, alphas = self.decoder(features, captions)
        return outputs, alphas

def load_checkpoint(model_path, vocab_path):
    if not os.path.exists(model_path) or not os.path.exists(vocab_path):
        return None, None, None
        
    # Load Dict 
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
        
    # Load Checkpoint with map_location
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize Model
    embed_size = 256
    hidden_size = 512
    vocab_size = len(vocab)
    attention_dim = 256
    
    model = ImageCaptioningModel(embed_size, hidden_size, vocab_size, attention_dim).to(device)
    
    # Load State Dict
    try:
        model.load_state_dict(checkpoint)
    except RuntimeError as e:
        print(f"Standard loading failed, trying to fix keys: {e}")
        # Try adjusting keys if needed (e.g. if saved from DataParallel)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            # Remove 'module.' prefix if present
            name = k.replace("module.", "")
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    model.eval()
    return model.encoder, model.decoder, vocab

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# --- Streamlit Interface ---

@st.cache_resource
def load_model():
    model_path = "best_model.pth"
    vocab_path = "vocab.pkl"
    
    # Reconstruct model if needed
    if not os.path.exists(model_path):
        # Check for chunks
        chunks = sorted([f for f in os.listdir(".") if f.startswith("model_chunk_")])
        if chunks:
            with open(model_path, "wb") as f_out:
                for chunk in chunks:
                    with open(chunk, "rb") as f_in:
                        f_out.write(f_in.read())
        else:
            return None, None, None

    encoder, decoder, vocab = load_checkpoint(model_path, vocab_path)
    return encoder, decoder, vocab

def main():
    st.set_page_config(page_title="Neural Storyteller", page_icon="🖼️")
    st.title("Neural Storyteller: Image Captioning 🖼️")
    st.markdown("Upload an image to generate a caption using an **Attention-based LSTM model**.")
    
    # Load model
    with st.spinner("Loading model..."):
        encoder, decoder, vocab = load_model()
    
    if encoder is None:
        st.error("Error: Model files (`best_model.pth`, `vocab.pkl`) not found. Please ensure they are in the same directory.")
        return

    # Sidebar for settings
    st.sidebar.header("Settings")
    method = st.sidebar.radio("Decoding Method", ["Beam Search", "Greedy"], index=0)
    
    # File Uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            if st.button("Generate Caption"):
                with st.spinner("Generating caption..."):
                    # Preprocess image
                    img_tensor = transform(image).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        features = encoder(img_tensor) # (1, 49, 2048)
                        
                        if method == "Beam Search":
                            caption_words = decoder.generate_caption_beam_search(features, vocab, beam_width=5)
                        else:
                            caption_words, _ = decoder.generate_caption(features, vocab)
                            
                    # Clean up caption
                    caption = ' '.join(caption_words)
                    if "<end>" in caption:
                        caption = caption.split("<end>")[0]
                    
                    st.success(f"**Caption:** {caption.strip()}")
            else:
                 st.info("Click the button to generate a caption.")

if __name__ == "__main__":
    main()
