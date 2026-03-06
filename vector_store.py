import os
import json
import numpy as np
from typing import List, Dict
import warnings

from test_cases import TestCases

warnings.filterwarnings('ignore')
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import requests
from config import CHUNK_SIZE, CHUNK_OVERLAP,OLLAMA_BASE_URL,EMBEDDING_MODEL,LLM_MODEL,CHUNK_SIZE,CHUNK_OVERLAP,TOP_K_RESULTS
from result_writer import ResultWriter

class VectorStore:
    """Basit in-memory vektör veritabanı"""

    def __init__(self):
        self.documents = []  # Orijinal metinler
        self.embeddings = np.array([])  # Embedding vektörleri
        self.metadata = []  # Chunk metadata

    def add_documents(self, chunks: List[Dict], embeddings: np.ndarray):
        """Chunk'ları ve embedding'leri ekle"""
        self.documents = [chunk['text'] for chunk in chunks]
        self.embeddings = embeddings
        self.metadata = chunks
        print(f"✓ {len(self.documents)} chunk ve embedding eklendi")

    def similarity_search(self, query_embedding: np.ndarray, k: int = 3) -> List[Dict]:
        """Benzerlik araması (Cosine Similarity)"""
        if len(self.embeddings) == 0:
            return []

        # Cosine similarity hesapla
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]

        # En benzer k dokument'i bul
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