import os
import json
import numpy as np
from typing import List, Dict
import warnings

from rag_pipeline import RagPipeline
from test_cases import TestCases

warnings.filterwarnings('ignore')
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import requests
from config import CHUNK_SIZE, CHUNK_OVERLAP,OLLAMA_BASE_URL,EMBEDDING_MODEL,LLM_MODEL,CHUNK_SIZE,CHUNK_OVERLAP,TOP_K_RESULTS
from result_writer import ResultWriter


print("="*70)
print("PDF SEMANTIC SEARCH CHATBOT")
print("Open Source Models (Ollama) + Cosine Similarity")
print("="*70)



def main():

    try:
        pipeline = RagPipeline()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return

    test_pdf_path = "test_document.pdf"

    if not os.path.exists(test_pdf_path):
        print("\n[WARNING] Test PDF not found. Place a PDF named 'test_document.pdf' in the folder.")
        return

    pipeline.load_pdf(test_pdf_path)

    test_cases = TestCases.get_test_cases()
    results = []

    for i, test in enumerate(test_cases, 1):
        print(f"\n[TEST {i}/{len(test_cases)}] {test['name']}")
        result = pipeline.query(test['question'], debug=True)
        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['sources']}")
        results.append({
            'test_name': test['name'],
            'question': test['question'],
            'answer': result['answer'],
            'similarity': result['sources'][0]['similarity'] if result['sources'] else 0,
            'test_type': test['expected_type']
        })

    results_df = pd.DataFrame(results)
    print("\nTEST SUMMARY")
    print(results_df[['test_name', 'similarity', 'test_type']])
    print(f"\nAverage Similarity Score: {results_df['similarity'].mean():.2%}")

    ResultWriter.save_results(results, results_df)

if __name__ == "__main__":
    main()