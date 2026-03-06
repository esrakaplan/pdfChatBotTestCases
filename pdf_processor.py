from PyPDF2 import PdfReader
from typing import List, Dict
from config import CHUNK_SIZE, CHUNK_OVERLAP

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