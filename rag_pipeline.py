import os
import json
import numpy as np
from typing import List, Dict
import warnings

from embedding_service import EmbeddingService
from llm_service import LLMService
from pdf_processor import PDFProcessor
from vector_store import VectorStore

warnings.filterwarnings('ignore')
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import requests
from config import CHUNK_SIZE, CHUNK_OVERLAP,OLLAMA_BASE_URL,EMBEDDING_MODEL,LLM_MODEL,CHUNK_SIZE,CHUNK_OVERLAP,TOP_K_RESULTS


class RagPipeline:
    """Retrieval Augmented Generation Pipeline"""

    def __init__(self):
        print("\n[BAŞLATMA] RAG Pipeline kuruluyor...")
        self.embedding_service = EmbeddingService(EMBEDDING_MODEL, OLLAMA_BASE_URL)
        self.llm_service = LLMService(LLM_MODEL, OLLAMA_BASE_URL)
        self.vector_store = VectorStore()
        print("✓ Pipeline hazır\n")

    def load_pdf(self, pdf_path: str):
        """PDF dosyasını yükle ve işle"""
        print(f"\n[PDF YÜKLEME] {pdf_path}")

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF bulunamadı: {pdf_path}")

        # Metni çıkar
        print("  Metin çıkarılıyor...")
        text = PDFProcessor.extract_text_from_pdf(pdf_path)
        print(f"  ✓ {len(text)} karakter çıkarıldı")

        # Chunk'lara böl
        print("  Metin chunk'lanıyor...")
        chunks = PDFProcessor.chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"  ✓ {len(chunks)} chunk oluşturuldu")

        # Embedding'ler üret
        print("  Embedding'ler üretiliyor...")
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedding_service.embed_texts(chunk_texts)

        # Vector store'a ekle
        print("  Vector store'a ekleniyor...")
        self.vector_store.add_documents(chunks, embeddings)

    def query(self, question: str, debug: bool = False) -> Dict:
        """Soruya cevap ver"""
        print(f"\n[SORU] {question}")

        # Sorunun embedding'ini üret
        query_embedding = self.embedding_service.embed_text(question)

        # Benzer chunk'ları bul
        search_results = self.vector_store.similarity_search(query_embedding, k=TOP_K_RESULTS)

        if not search_results:
            return {
                'query': question,
                'answer': 'Dokümanda ilgili bilgi bulunamadı',
                'sources': [],
                'debug': {}
            }

        # Context'i birleştir
        context = "\n\n---\n\n".join([
            f"[Chunk {r['chunk_id']} - Benzerlik: {r['similarity']:.2%}]\n{r['text']}"
            for r in search_results
        ])

        # LLM ile cevap üret
        print("[CEVAP ÜRETİLİYOR]")
        answer = self.llm_service.generate_answer(question, context)

        result = {
            'query': question,
            'answer': answer,
            'sources': [
                {
                    'chunk_id': r['chunk_id'],
                    'similarity': round(r['similarity'], 4),
                    'length': r['length']
                }
                for r in search_results
            ],
            'debug': {
                'retrieved_chunks': len(search_results),
                'top_similarity': search_results[0]['similarity']
            } if debug else {}
        }

        return result