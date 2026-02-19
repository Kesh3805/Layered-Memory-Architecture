# embeddings.py
import numpy as np
from sentence_transformers import SentenceTransformer

# Load a lightweight local embedding model (no API key needed)
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text: str):
    # Returns 384-dimensional embedding
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.astype('float32')
