import requests
import numpy as np
from typing import List

class EmbeddingService:
    """Ollama ile embedding üretme"""

    def __init__(self, model_name: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.check_model()

    def check_model(self):
        """Model'in çalışıp çalışmadığını kontrol et"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model_name, "prompt": "test", "stream": False},
                timeout=50
            )
            if response.status_code != 200:
                print(f"⚠️  {self.model_name} modeli başlatılıyor... (ilk kez yavaş olabilir)")
        except Exception as e:
            raise Exception(
                f"Ollama bağlantı hatası! Ollama'yı çalıştırdığınızdan emin olun:\n"
                f"Terminal'de: ollama serve\n"
                f"Hata: {e}"
            )

    def embed_text(self, text: str) -> np.ndarray:
        """Metin için embedding üret"""
        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model_name, "input": text},
                timeout=60
            )

            if response.status_code != 200:
                raise Exception(f"Embedding hatası: {response.text}")

            data = response.json()
            return np.array(data['embeddings'][0])
        except Exception as e:
            raise Exception(f"Embedding üretme hatası: {e}")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Birden fazla metin için embedding üret"""
        embeddings = []
        for i, text in enumerate(texts):
            if i % 10 == 0:
                print(f"  Embedding üretiliyor: {i}/{len(texts)}")
            embeddings.append(self.embed_text(text))
        return np.array(embeddings)
