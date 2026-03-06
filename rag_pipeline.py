import os
from typing import Dict
import warnings
from embedding_service import EmbeddingService
from llm_service import LLMService
from pdf_processor import PDFProcessor
from vector_store import VectorStore
warnings.filterwarnings('ignore')

from config import OLLAMA_BASE_URL,EMBEDDING_MODEL,LLM_MODEL,CHUNK_SIZE,CHUNK_OVERLAP,TOP_K_RESULTS


class RagPipeline:
    """Retrieval Augmented Generation Pipeline"""

    def __init__(self):
        print("\n[STARTING] RAG Pipeline is being installed...")
        self.embedding_service = EmbeddingService(EMBEDDING_MODEL, OLLAMA_BASE_URL)
        self.llm_service = LLMService(LLM_MODEL, OLLAMA_BASE_URL)
        self.vector_store = VectorStore()
        print("✓ Pipeline ready\n")

    def load_pdf(self, pdf_path: str):
        print(f"\n[PDF LOAD] {pdf_path}")

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        print("Extracting text...")
        text = PDFProcessor.extract_text_from_pdf(pdf_path)
        print(f" ✓ {len(text)} characters removed")

        print(" Chunking text...")
        chunks = PDFProcessor.chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f" ✓ {len(chunks)} chunk created")

        # Generate Embeddings
        print("Embeddings are being produced...")
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedding_service.embed_texts(chunk_texts)

        # Add to Vector store
        print("Adding to vector store...")
        self.vector_store.add_documents(chunks, embeddings)

    def query(self, question: str, debug: bool = False) -> Dict:
        print(f"\n[QUESTION] {question}")

        query_embedding = self.embedding_service.embed_text(question)

        search_results = self.vector_store.similarity_search(query_embedding, k=TOP_K_RESULTS)

        if not search_results:
            return {
                'query': question,
                'answer': 'No relevant information found in the document',
                'sources': [],
                'debug': {}
            }

        context = "\n\n---\n\n".join([
            f"[Chunk {r['chunk_id']} - Similarity: {r['similarity']:.2%}]\n{r['text']}"
            for r in search_results
        ])

        print("[PRODUCING ANSWER]")
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