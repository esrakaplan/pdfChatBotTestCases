import os
import json
import numpy as np
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import requests



print("="*70)
print("PDF SEMANTIC SEARCH CHATBOT")
print("Açık Kaynak Modeller (Ollama) + Cosine Similarity")
print("="*70)


OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "mistral:latest"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 3

# ===== 1. PDF'DEN METİN ÇIKAR =====
class PDFProcessor:
    """PDF dosyasından metin çıkarma"""

    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """PDF'den metin çıkar"""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += f"\n[SAYFA {page_num + 1}]\n"
                text += page.extract_text()
            return text
        except Exception as e:
            raise Exception(f"PDF okuma hatası: {e}")

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[Dict]:
        """Metni chunklara böl (RAG için)"""
        chunks = []

        # Paragraflarla böl
        paragraphs = text.split('\n\n')

        current_chunk = ""
        chunk_id = 0

        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) < chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk.strip():
                    chunks.append({
                        'id': chunk_id,
                        'text': current_chunk.strip(),
                        'length': len(current_chunk)
                    })
                    chunk_id += 1

                # Overlap ile başla
                current_chunk = paragraph + "\n\n"

        # Son chunk'ı ekle
        if current_chunk.strip():
            chunks.append({
                'id': chunk_id,
                'text': current_chunk.strip(),
                'length': len(current_chunk)
            })

        return chunks

# ===== 2. EMBEDDING VE VEKTÖR STORE =====
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

# ===== 3. VEKTÖR VERİTABANI (IN-MEMORY) =====
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

# ===== 4. LLM ile CEVAP ÜRETİMİ =====
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

# ===== 5. RAG PIPELINE =====
class RAG_Pipeline:
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

# ===== 6. TEST VE EVALUATION =====
class TestCases:
    """Test cases ve edge cases"""
    
    @staticmethod
    def get_test_cases() -> List[Dict]:
        """Test case'leri döndür"""
        return [
            {
                'name': 'Temel Soru',
                'question': 'Bu dokümanda ana konu nedir?',
                'expected_type': 'summary',
                'description': 'Dokümandaki ana konuyu anlaması beklenir'
            }
        ]

# ===== 7. DEMO VE ÇALIŞMA =====
def main():
    """Ana fonksiyon"""
    
    # RAG Pipeline'ı başlat
    try:
        pipeline = RAG_Pipeline()
    except Exception as e:
        print(f"\n❌ HATA: {e}")
        print("\nOllama kurulumunu kontrol edin:")
        print("1. https://ollama.ai adresinden Ollama'yı indirin")
        print("2. Terminal'de: ollama pull nomic-embed-text")
        print("3. Terminal'de: ollama pull mistral")
        print("4. Terminal'de: ollama serve")
        return
    
    # Test PDF oluştur
    test_pdf_path = "test_document.pdf"
    
    # Eğer test PDF yoksa demo metin oluştur
    if not os.path.exists(test_pdf_path):
        print("\n[UYARI] Test PDF bulunamadı")
        print("Lütfen bir PDF dosyasını 'test_document.pdf' olarak aynı dizine koyun")
        print("\nDemo için örnek test verisi oluşturuluyor...")
        
        # Demo veri
        demo_text = """
        YAPAY ZEKA VE MAKINE ÖĞRENMESİ

        Yapay zeka (AI), bilgisayarların insan benzeri görevleri yerine getirme yeteneğidir.
        
        Makine öğrenmesi (ML), yapay zekanın bir alt alanıdır ve verilerden kalıpları öğrenmeyi sağlar.
        
        Derin öğrenme, sinir ağları kullanarak çok karmaşık kalıpları öğrenebilir.
        
        Natural Language Processing (NLP), bilgisayarların doğal dil anlaması sağlar.
        
        Sınıflandırma (Classification), verileri kategorilere ayırmak için kullanılır.
        
        Regresyon (Regression), sürekli değerleri tahmin etmek için kullanılır.
        
        Kümeleme (Clustering), benzer verileri grup haline getirmek için kullanılır.
        
        Denetimsiz öğrenme, etiketli veri olmadan kalıplar öğrenir.
        
        Denetimli öğrenme, etiketli verilerle modelini eğitir.
        
        2023 yılında büyük dil modelleri (LLM) önemli gelişmeler göstermiştir.
        """
        
        # Mock embedding ve store (gerçek PDF olmadan test yapabilmek için)
        print("\n⚠️  GERÇEK PDF OLMADAN DEMO MODU")
        print("Bu demo sadece sistem kontrolü içindir")
        return
    
    # PDF yükle
    try:
        pipeline.load_pdf(test_pdf_path)
    except Exception as e:
        print(f"❌ PDF yükleme hatası: {e}")
        return
    
    # Test cases
    print("\n" + "="*70)
    print("TEST CASELER VE EDGE CASELER")
    print("="*70)
    
    test_cases = TestCases.get_test_cases()
    
    results = []
    for i, test in enumerate(test_cases, 1):
        print(f"\n[TEST {i}/{len(test_cases)}] {test['name']}")
        print(f"Soru: {test['question']}")
        print(f"Beklenen: {test['description']}")
        print("-" * 70)
        
        result = pipeline.query(test['question'], debug=True)
        
        print(f"\nCevap: {result['answer']}")
        print(f"\nKaynak: {result['sources']}")
        
        results.append({
            'test_name': test['name'],
            'question': test['question'],
            'answer': result['answer'],
            'similarity': result['sources'][0]['similarity'] if result['sources'] else 0,
            'test_type': test['expected_type']
        })
    
    # Özet rapor
    print("\n" + "="*70)
    print("TEST ÖZETİ")
    print("="*70)
    
    results_df = pd.DataFrame(results)
    print("\nTest Sonuçları:")
    print(results_df[['test_name', 'similarity', 'test_type']])
    
    # İstatistikler
    print(f"\nOrtalama Benzerlik Skoru: {results_df['similarity'].mean():.2%}")
    print(f"Toplam Test: {len(results_df)}")
    
    # Cevapları kaydet
    output_file = 'chatbot_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_tests': len(results),
            'results': results,
            'summary': {
                'avg_similarity': float(results_df['similarity'].mean()),
                'timestamp': pd.Timestamp.now().isoformat()
            }
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Sonuçlar '{output_file}' dosyasına kaydedildi")

if __name__ == "__main__":
    main()
