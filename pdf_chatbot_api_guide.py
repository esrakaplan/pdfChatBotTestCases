"""
PDF CHATBOT - BEDAVA API'LER KARŞILAŞTIRMASI

Seçenek 1: GROQ (En Hızlı - Bedava)
- Ücretsiz API key: https://console.groq.com
- LLM Model: mixtral-8x7b-32768 (çok hızlı)
- Embedding: Hugging Face veya Ollama

Seçenek 2: Hugging Face (Bedava)
- Inference API: https://huggingface.co/inference-api
- Modeller: mistral-7b, zephyr-7b-beta, Llama-2-7b-chat
- Embedding: sentence-transformers/all-MiniLM-L6-v2

Seçenek 3: Ollama + LocalAI (Tamamen Lokal - Sınırsız, Bedava)
- Indirin: https://ollama.ai
- Modeller: mistral, neural-chat, orca-mini
- Embedding: nomic-embed-text

Seçenek 4: Google Colab + Transformers (Bedava GPU)
- Runtime: T4 GPU
- Modeller: türü model indirebilirsiniz

ÖNERİ: Başlamak için -> Ollama (lokal, sınırsız)
       Hızlı test için -> Groq (çok hızlı, bedava)
       Üretim için -> Ollama + Open Source Models
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# PDF işleme
try:
    from PyPDF2 import PdfReader
except ImportError:
    print("pip install PyPDF2")

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# API istekleri
import requests
from dotenv import load_dotenv

load_dotenv()

print("="*80)
print("PDF SEMANTIC SEARCH CHATBOT - BEDAVA API'LER")
print("="*80)

# ===== API SEÇENEKLERİ =====
class APIProvider(Enum):
    """Desteklenen API sağlayıcıları"""
    OLLAMA = "ollama"  # Lokal
    GROQ = "groq"      # Bedava çok hızlı
    HUGGINGFACE = "huggingface"  # Bedava modeller
    COLAB = "colab"    # Google Colab

# ===== 1. PDF İŞLEME =====
class PDFProcessor:
    """PDF dosyasından metin çıkarma"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> Tuple[str, List[str]]:
        """PDF'den metin ve sayfa bilgilerini çıkar"""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            pages = []
            
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                page_text = page.extract_text()
                text += f"\n[SAYFA {page_num + 1}]\n{page_text}\n"
                pages.append(page_text)
            
            return text, pages
        except Exception as e:
            raise Exception(f"PDF okuma hatası: {e}")
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
        """Metni chunk'lara böl"""
        chunks = []
        sentences = text.split('. ')
        
        current_chunk = ""
        chunk_id = 0
        
        for sentence in sentences:
            sentence = sentence.strip() + '. ' if sentence.strip() else ''
            
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence
            else:
                if current_chunk.strip():
                    chunks.append({
                        'id': chunk_id,
                        'text': current_chunk.strip(),
                        'tokens': len(current_chunk.split()),
                        'length': len(current_chunk)
                    })
                    chunk_id += 1
                current_chunk = sentence
        
        if current_chunk.strip():
            chunks.append({
                'id': chunk_id,
                'text': current_chunk.strip(),
                'tokens': len(current_chunk.split()),
                'length': len(current_chunk)
            })
        
        return chunks

# ===== 2. EMBEDDING SERVISLERI =====
class EmbeddingService:
    """Farklı embedding servisleri"""
    
    def __init__(self, provider: APIProvider = APIProvider.OLLAMA):
        self.provider = provider
        self.model_name = "nomic-embed-text" if provider == APIProvider.OLLAMA else "sentence-transformers/all-MiniLM-L6-v2"
    
    def embed_text(self, text: str) -> np.ndarray:
        """Metin için embedding üret"""
        if self.provider == APIProvider.OLLAMA:
            return self._embed_ollama(text)
        elif self.provider == APIProvider.HUGGINGFACE:
            return self._embed_huggingface(text)
        else:
            raise NotImplementedError(f"{self.provider} desteklenmiyor")
    
    def _embed_ollama(self, text: str) -> np.ndarray:
        """Ollama embedding"""
        try:
            response = requests.post(
                "http://localhost:11434/api/embed",
                json={"model": "nomic-embed-text", "input": text},
                timeout=30
            )
            return np.array(response.json()['embeddings'][0])
        except Exception as e:
            raise Exception(f"Ollama embedding hatası: {e}")
    
    def _embed_huggingface(self, text: str) -> np.ndarray:
        """Hugging Face Inference API embedding"""
        api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not api_key:
            raise Exception("HUGGINGFACE_API_KEY environment variable ayarlanmadı")
        
        try:
            response = requests.post(
                "https://api-inference.huggingface.co/pipeline/feature-extraction",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"inputs": text},
                timeout=30
            )
            return np.array(response.json())
        except Exception as e:
            raise Exception(f"Hugging Face embedding hatası: {e}")
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Birden fazla metin embedding"""
        embeddings = []
        for i, text in enumerate(texts):
            if i % 10 == 0:
                print(f"  Embedding {i}/{len(texts)}")
            embeddings.append(self.embed_text(text))
        return np.array(embeddings)

# ===== 3. LLM SERVISLERI =====
class LLMService:
    """Farklı LLM servisleri"""
    
    def __init__(self, provider: APIProvider = APIProvider.OLLAMA):
        self.provider = provider
    
    def generate_answer(self, query: str, context: str) -> str:
        """Context'e dayanarak cevap üret"""
        if self.provider == APIProvider.OLLAMA:
            return self._generate_ollama(query, context)
        elif self.provider == APIProvider.GROQ:
            return self._generate_groq(query, context)
        elif self.provider == APIProvider.HUGGINGFACE:
            return self._generate_huggingface(query, context)
        else:
            raise NotImplementedError(f"{self.provider} desteklenmiyor")
    
    def _generate_ollama(self, query: str, context: str) -> str:
        """Ollama ile cevap üret"""
        prompt = f"""Context:
{context}

Soru: {query}

Cevap (kısa ve öz):"""
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "mistral",
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.3
                },
                timeout=60
            )
            return response.json().get('response', 'Hata').strip()
        except Exception as e:
            return f"Ollama hatası: {e}"
    
    def _generate_groq(self, query: str, context: str) -> str:
        """Groq API ile cevap üret (Bedava ve çok hızlı!)"""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise Exception("GROQ_API_KEY environment variable ayarlanmadı")
        
        prompt = f"""Context:
{context}

Soru: {query}

Kısa ve açık bir cevap ver (maksimum 2-3 cümle):"""
        
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "mixtral-8x7b-32768",
                    "messages": [
                        {"role": "system", "content": "Sen yardımcı bir asistanssın. Verilen context'e dayanarak sorulara cevap ver."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 500
                },
                timeout=30
            )
            
            if response.status_code != 200:
                return f"Groq hatası: {response.text}"
            
            return response.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            return f"Groq hatası: {e}"
    
    def _generate_huggingface(self, query: str, context: str) -> str:
        """Hugging Face Inference API ile cevap üret"""
        api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not api_key:
            raise Exception("HUGGINGFACE_API_KEY environment variable ayarlanmadı")
        
        prompt = f"""Context:
{context}

Soru: {query}

Cevap:"""
        
        try:
            response = requests.post(
                "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"inputs": prompt, "parameters": {"max_length": 512}},
                timeout=30
            )
            
            if response.status_code != 200:
                return f"Hugging Face hatası: {response.text}"
            
            return response.json()[0]['generated_text'].strip()
        except Exception as e:
            return f"Hugging Face hatası: {e}"

# ===== 4. VEKTÖR STORE =====
class VectorStore:
    """In-memory vektör veritabanı"""
    
    def __init__(self):
        self.documents = []
        self.embeddings = np.array([])
        self.metadata = []
    
    def add_documents(self, chunks: List[Dict], embeddings: np.ndarray):
        """Chunk'ları ekle"""
        self.documents = [chunk['text'] for chunk in chunks]
        self.embeddings = embeddings
        self.metadata = chunks
        print(f"✓ {len(self.documents)} chunk eklendi")
    
    def similarity_search(self, query_embedding: np.ndarray, k: int = 3) -> List[Dict]:
        """Benzerlik araması"""
        if len(self.embeddings) == 0:
            return []
        
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'text': self.documents[idx],
                'similarity': float(similarities[idx]),
                'chunk_id': self.metadata[idx]['id']
            })
        
        return results

# ===== 5. RAG PIPELINE =====
class RAG_Pipeline:
    """Retrieval Augmented Generation"""
    
    def __init__(self, provider: APIProvider = APIProvider.OLLAMA):
        print(f"\n[BAŞLATMA] RAG Pipeline ({provider.value})")
        self.provider = provider
        self.embedding_service = EmbeddingService(provider)
        self.llm_service = LLMService(provider)
        self.vector_store = VectorStore()
    
    def load_pdf(self, pdf_path: str):
        """PDF yükle ve işle"""
        print(f"\n[PDF YÜKLEME] {pdf_path}")
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF bulunamadı: {pdf_path}")
        
        # Metni çıkar
        text, pages = PDFProcessor.extract_text_from_pdf(pdf_path)
        print(f"  ✓ {len(text)} karakter, {len(pages)} sayfa")
        
        # Chunk'la
        chunks = PDFProcessor.chunk_text(text)
        print(f"  ✓ {len(chunks)} chunk oluşturuldu")
        
        # Embedding'ler
        print("  Embedding'ler üretiliyor...")
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedding_service.embed_texts(chunk_texts)
        
        # Store'a ekle
        self.vector_store.add_documents(chunks, embeddings)
    
    def query(self, question: str) -> Dict:
        """Soruya cevap ver"""
        print(f"\n[SORU] {question}")
        
        # Embedding
        query_embedding = self.embedding_service.embed_text(question)
        
        # Arama
        results = self.vector_store.similarity_search(query_embedding, k=3)
        
        if not results:
            return {'query': question, 'answer': 'Bilgi bulunamadı', 'sources': []}
        
        # Context
        context = "\n\n".join([
            f"[Chunk {r['chunk_id']}]\n{r['text']}"
            for r in results
        ])
        
        # LLM cevabı
        answer = self.llm_service.generate_answer(question, context)
        
        return {
            'query': question,
            'answer': answer,
            'sources': results,
            'provider': self.provider.value
        }

# ===== 6. TEST CASE'LER =====
class TestCases:
    """Test ve edge case'ler"""
    
    @staticmethod
    def get_test_cases() -> List[Dict]:
        return [
            {'q': 'Bu belge hakkında kısaca bilgi verir misin?', 'type': 'summary'},
            {'q': 'Belgede en önemli noktalar nelerdir?', 'type': 'key_points'},
            {'q': 'Bu konuyla ilgili tanımlar var mı?', 'type': 'definitions'},
            {'q': 'Ne?', 'type': 'edge_case_short'},
            {'q': 'Bu belgede hiç anlatılmayan konu nedir?', 'type': 'hallucination'},
        ]

# ===== 7. KARŞILAŞTIRMA FONKSIYONU =====
def compare_api_providers(pdf_path: str, test_question: str):
    """Farklı API'leri karşılaştır"""
    
    providers = [
        APIProvider.OLLAMA,
        APIProvider.GROQ,
        APIProvider.HUGGINGFACE
    ]
    
    results = []
    
    for provider in providers:
        print(f"\n{'='*80}")
        print(f"TESTING: {provider.value.upper()}")
        print(f"{'='*80}")
        
        try:
            pipeline = RAG_Pipeline(provider=provider)
            pipeline.load_pdf(pdf_path)
            result = pipeline.query(test_question)
            
            results.append({
                'provider': provider.value,
                'status': 'success',
                'answer_length': len(result['answer']),
                'top_similarity': result['sources'][0]['similarity'] if result['sources'] else 0
            })
            
            print(f"✓ Başarılı")
            print(f"Cevap: {result['answer'][:100]}...")
        except Exception as e:
            print(f"❌ Hata: {e}")
            results.append({
                'provider': provider.value,
                'status': 'error',
                'error': str(e)
            })
    
    # Karşılaştırma tablosu
    print(f"\n{'='*80}")
    print("API KARŞILAŞTIRMASI")
    print(f"{'='*80}\n")
    
    df = pd.DataFrame(results)
    print(df)
    
    return results

# ===== 8. SETUP REHBERI =====
def print_setup_guide():
    """Kurulum rehberi"""
    
    setup_text = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                  PDF CHATBOT - API KURULUM REHBERİ                          ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ SEÇENEK 1: OLLAMA (ÖNERILEN - TAMAMEN LOKAL, SINIRSIZ, BEDAVA)            │
    └─────────────────────────────────────────────────────────────────────────────┘
    
    1. Ollama'yı indirin: https://ollama.ai
    2. Kurulum tamamlandıktan sonra terminal açın:
       
       ollama pull nomic-embed-text      # Embedding modeli
       ollama pull mistral                # LLM modeli
    
    3. Ollama'yı başlatın (arka planda çalışır):
       
       ollama serve
    
    4. Script'i çalıştırın:
       
       python pdf_chatbot_ollama.py
    
    Avantajları:
    ✓ Tamamen lokal çalışır (internet gerekli değil)
    ✓ Sınırsız sorgu yapabilirsiniz
    ✓ Hiçbir ücret yok
    ✓ Veriniz özel kalır
    
    Dezavantajları:
    ✗ İlk yükleme biraz zaman alır (~2-3 GB indir)
    ✗ CPU/GPU kullanır (ancak eski bilgisayarlarda da çalışır)
    
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ SEÇENEK 2: GROQ (BEDAVA, ÇOK HIZLI - İNTERNET GEREKLI)                    │
    └─────────────────────────────────────────────────────────────────────────────┘
    
    1. https://console.groq.com adresine gidin
    2. Ücretsiz hesap oluşturun
    3. API key'inizi kopyalayın
    4. .env dosyası oluşturun:
       
       GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    
    5. Script'i çalıştırın:
       
       python pdf_chatbot_api.py --provider groq
    
    Avantajları:
    ✓ Çok hızlı cevaplar
    ✓ Kurulum kolay
    ✓ Bedava ve sınırsız
    
    Dezavantajları:
    ✗ İnternet bağlantısı gerekli
    ✗ API rate limit'i var (saatte 100 request vb.)
    
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ SEÇENEK 3: HUGGING FACE (BEDAVA - İNTERNET GEREKLI)                       │
    └─────────────────────────────────────────────────────────────────────────────┘
    
    1. https://huggingface.co adresine gidin
    2. Ücretsiz hesap oluşturun
    3. API token'ınızı alın
    4. .env dosyası oluşturun:
       
       HUGGINGFACE_API_KEY=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    
    5. Script'i çalıştırın:
       
       python pdf_chatbot_api.py --provider huggingface
    
    Avantajları:
    ✓ Çok sayıda model seçeneği
    ✓ Bedava
    ✓ Açık kaynak modeller
    
    Dezavantajları:
    ✗ Bazen yavaş olabilir
    ✗ İnternet bağlantısı gerekli
    
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ ÖNERİ: HANGİSİ KULLANMALIYIM?                                              │
    └─────────────────────────────────────────────────────────────────────────────┘
    
    Başlamak için (basit proje):
    → OLLAMA (en iyi seçenek, sorunsuz)
    
    Hızlı test yapmak istiyorsan:
    → GROQ (1 dakikada kurulur, çok hızlı)
    
    Üretim ortamı (100+ dokuman):
    → OLLAMA + Mistral/Llama2 (veya Groq API)
    
    Maliyet hesabı (100 dokuman x 1000 sorgu = 100.000 sorgu):
    - Ollama: 0 TL (lokal)
    - Groq: 0 TL (bedava)
    - OpenAI: 2000+ TL (çok pahalı)
    
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ .env DOSYASI ÖRNEĞI                                                        │
    └─────────────────────────────────────────────────────────────────────────────┘
    
    # Groq API Key
    GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    
    # Hugging Face API Key
    HUGGINGFACE_API_KEY=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    
    # Ollama (lokal, .env'ye gerek yok)
    OLLAMA_BASE_URL=http://localhost:11434
    """
    
    print(setup_text)

# ===== MAIN =====
if __name__ == "__main__":
    
    # Kurulum rehberi
    print_setup_guide()
    
    # Örnek kullanım
    print("\n" + "="*80)
    print("ÖRNEK KULLANIM")
    print("="*80 + "\n")
    
    print("1. PDF dosyasını hazırlayın (test_document.pdf)")
    print("2. API key'inizi .env dosyasına yazın (Ollama için gerek yok)")
    print("3. Script'i çalıştırın:\n")
    
    print("   # Ollama ile")
    print("   python pdf_chatbot_api.py --provider ollama\n")
    
    print("   # Groq ile")
    print("   python pdf_chatbot_api.py --provider groq\n")
    
    print("   # Karşılaştırma")
    print("   python pdf_chatbot_api.py --compare\n")
