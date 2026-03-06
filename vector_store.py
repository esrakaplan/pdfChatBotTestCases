import numpy as np
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics.pairwise import cosine_similarity


class VectorStore:
    def __init__(self):
        self.documents = []
        self.embeddings = np.array([])
        self.metadata = []  # Chunk metadata

    def add_documents(self, chunks: List[Dict], embeddings: np.ndarray):
        self.documents = [chunk['text'] for chunk in chunks]
        self.embeddings = embeddings
        self.metadata = chunks
        print(f"✓ {len(self.documents)} chunk & embedding are added")

    def similarity_search(self, query_embedding: np.ndarray, k: int = 3) -> List[Dict]:
        if len(self.embeddings) == 0:
            return []

        similarities = cosine_similarity([query_embedding], self.embeddings)[0]

        top_indices = np.argsort(similarities)[-k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                'text': self.documents[idx],
                'similarity': float(similarities[idx]),
                'chunk_id': self.metadata[idx]['id'],
                'length': self.metadata[idx]['length']
            })

        return results