# vector_store.py
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

class PlagiarismVectorDB:
    def __init__(self):
        self.index = None
        self.texts = []

    def build_index(self, reference_texts):
        """Build FAISS index from a list of reference documents."""
        self.texts = reference_texts
        embeddings = model.encode(reference_texts, convert_to_numpy=True)
        dimension = embeddings.shape[1]

        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(dimension)  # cosine similarity (inner product)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

    def query(self, input_text, top_k=5):
        """Search for top_k most similar documents."""
        input_embedding = model.encode([input_text], convert_to_numpy=True)
        faiss.normalize_L2(input_embedding)
        distances, indices = self.index.search(input_embedding, top_k)

        results = []
        for score, idx in zip(distances[0], indices[0]):
            results.append({"reference": self.texts[idx], "similarity": round(float(score) * 100, 2)})
        return results

# Create a global instance
plagiarism_db = PlagiarismVectorDB()
