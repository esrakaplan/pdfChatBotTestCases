import requests
import numpy as np
from typing import List

from config import OLLAMA_BASE_URL, EMBEDDING_MODEL, MODEL_CONN_TIMEOUT


class EmbeddingService:

    def __init__(self, model_name: str = EMBEDDING_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model_name = model_name
        self.base_url = base_url
        self.check_model()

    def check_model(self):
        """Check if the model is running"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model_name, "prompt": "test", "stream": False},
                timeout=MODEL_CONN_TIMEOUT
            )
            if response.status_code != 200:
                print(f"⚠️  {self.model_name} model is starting... (may be slow on first run)")
        except Exception as e:
            raise Exception(
                f"Ollama connection error! Make sure Ollama is running:\n"
                f"Error: {e}"
            )

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model_name, "input": text},
                timeout=60
            )

            if response.status_code != 200:
                raise Exception(f"Embedding error: {response.text}")

            data = response.json()
            return np.array(data['embeddings'][0])
        except Exception as e:
            raise Exception(f"Embedding generation error: {e}")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        embeddings = []
        for i, text in enumerate(texts):
            if i % 10 == 0:
                print(f"  Generating embeddings: {i}/{len(texts)}")
            embeddings.append(self.embed_text(text))
        return np.array(embeddings)