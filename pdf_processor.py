from PyPDF2 import PdfReader
from typing import List, Dict
from config import CHUNK_SIZE, CHUNK_OVERLAP

class PDFProcessor:

    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += f"\n[Page {page_num + 1}]\n"
                text += page.extract_text()
            return text
        except Exception as e:
            raise Exception(f"PDF read exception: {e}")

    @staticmethod
    def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict]:
        chunks = []

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

                current_chunk = paragraph + "\n\n"

        if current_chunk.strip():
            chunks.append({
                'id': chunk_id,
                'text': current_chunk.strip(),
                'length': len(current_chunk)
            })

        return chunks