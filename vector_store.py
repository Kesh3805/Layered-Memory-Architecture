# vector_store.py
import faiss
import numpy as np
import pickle
from embeddings import get_embedding

dimension = 384  # all-MiniLM-L6-v2 produces 384-dim embeddings
index = faiss.IndexFlatL2(dimension)
documents = []

def add_documents(text_chunks):
    global documents
    vectors = np.array([get_embedding(t) for t in text_chunks]).astype("float32")
    index.add(vectors)
    documents.extend(text_chunks)

def search(query, k=4):
    query_vector = get_embedding(query).astype("float32").reshape(1, -1)
    distances, indices = index.search(query_vector, k)
    return [documents[i] for i in indices[0]]
