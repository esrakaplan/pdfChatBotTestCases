import os
import json
import numpy as np
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import requests
from config import CHUNK_SIZE, CHUNK_OVERLAP,OLLAMA_BASE_URL,EMBEDDING_MODEL,LLM_MODEL,CHUNK_SIZE,CHUNK_OVERLAP,TOP_K_RESULTS


class LLMService:
    """Ollama ile LLM cevapları üret"""

    def __init__(self, model_name: str = "mistral", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url

    def generate_answer(self, query: str, context: str, max_tokens: int = 300) -> str:
        """Context'e dayanarak cevap üret"""

        prompt = f"""Sana bir soru ve context metni verilecek. 
Context'te bulduğun bilgilere dayanarak soru'ya cevap ver.

Context:
{context}

Soru: {query}

Cevap:"""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.3  # Daha tutarlı cevaplar
                },
                timeout=180
            )

            if response.status_code != 200:
                return f"Hata: {response.text}"

            data = response.json()
            return data.get('response', 'Cevap üretilemiyor').strip()
        except Exception as e:
            return f"LLM Hata: {e}"