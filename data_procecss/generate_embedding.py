"""
Scientific Document Embedding Pipeline using Pre-trained SciBERT Model
This code implements a feature extraction pipeline for generating contextual embeddings
from research paper abstracts using the SciBERT language model.
Reference: Beltagy et al. (2019). SciBERT: A Pretrained Language Model for Scientific Text. EMNLP.
"""
import torch
import pandas as pd
import numpy as np
import csv
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import AutoTokenizer, AutoModel

# Model Configuration Section --------------------------------------------------
MODEL_NAME = 'allenai/scibert_scivocab_uncased'  # Official SciBERT variant trained on scientific corpus
# MODEL_NAME = "AyoubChLin/bert-finetuned-Arxiv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Hardware optimization

# Initialize model components with proper error handling
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    print(f"Successfully initialized SciBERT model on {DEVICE}")
except Exception as e:
    print(f"Model initialization failed: {str(e)}")
    exit(1)

# Data Configuration -----------------------------------------------------------
CSV_PATH = 'papers.csv'  # Input CSV containing paper metadata
REQUIRED_COLUMNS = ['paper_id', 'abstract']  # Mandatory data columns

# Data Validation and Preprocessing
try:
    data = pd.read_csv(CSV_PATH)
    assert all(col in data.columns for col in REQUIRED_COLUMNS), "Missing required columns"

    # Text normalization pipeline
    data['abstract'] = (data['abstract']
                        .fillna('')  # Handle missing values
                        .astype(str)  # Ensure string type
                        .str.strip())  # Remove leading/trailing whitespace
    print(f"Loaded {len(data)} papers with abstracts")
except FileNotFoundError:
    print(f"Data file not found: {CSV_PATH}")
    exit(1)

# Embedding Generation Pipeline ------------------------------------------------
EMBEDDING_METHODS = {
    'cls_token': lambda x: x.last_hidden_state[:, 0, :],  # CLS token embedding
    'mean_pooling': lambda x: x.last_hidden_state.mean(dim=1)  # Mean pooling strategy
}

# Initialize storage structures
embeddings_records = []  # For tabular storage (CSV)
embeddings_lookup = {}  # For efficient retrieval (dictionary)

# Batch processing parameters
MAX_SEQ_LENGTH = 512  # Transformer model's maximum sequence length
TRUNCATION = True  # Enable text truncation for long abstracts
PADDING = 'max_length'  # Standardize input length

for _, row in data.iterrows():
    paper_id = row['paper_id']
    abstract_text = row['abstract']

    # Tokenization with scientific text considerations
    inputs = tokenizer(
        abstract_text,
        return_tensors="pt",
        max_length=MAX_SEQ_LENGTH,
        truncation=TRUNCATION,
        padding=PADDING
    ).to(DEVICE)

    # Feature extraction with gradient computation disabled
    with torch.no_grad():
        model_output = model(**inputs)

    # Generate multiple embedding variants
    cls_embedding = EMBEDDING_METHODS['cls_token'](model_output).cpu().numpy().astype(np.float32)
    mean_embedding = EMBEDDING_METHODS['mean_pooling'](model_output).squeeze().cpu().numpy().astype(np.float32)

    # Store embeddings with multiple persistence strategies
    embeddings_records.append({
        "paper_id": paper_id,
        "cls_embedding": cls_embedding.tolist(),  # JSON-serializable format
        "mean_embedding": mean_embedding.tolist()
    })

    embeddings_lookup[paper_id] = {
        'cls': cls_embedding,
        'mean': mean_embedding
    }

# Data Persistence Section -----------------------------------------------------
# Save to CSV for human-readable format
(pd.DataFrame(embeddings_records)
 .to_csv('embedding/papers_data_embed.csv', index=False, encoding='utf-8'))

# Save to NPY for efficient numerical storage
np.save('embedding/embeddings_lookup.npy', embeddings_lookup)

print(f"Successfully processed {len(embeddings_records)} papers")
print(f"Embedding dimensions - CLS: {cls_embedding.shape}, Mean: {mean_embedding.shape}")