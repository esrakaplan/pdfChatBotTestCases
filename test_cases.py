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
from config import CHUNK_SIZE, CHUNK_OVERLAP, OLLAMA_BASE_URL, EMBEDDING_MODEL, LLM_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, \
    TOP_K_RESULTS




class TestCases:
    """Provide test cases for evaluation"""

    @staticmethod
    def get_test_cases() -> List[Dict]:
        return [
            {
                'name': 'Basic Question',
                'question': 'What is the main topic of this document?',
                'expected_type': 'summary',
                'description': 'Should summarize the main topic from the document'
            }
        ]
