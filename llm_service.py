import warnings
warnings.filterwarnings('ignore')

import requests
from config import OLLAMA_BASE_URL, LLM_MODEL, READ_TIMEOUT


class LLMService:
    """Create LLM response with Ollama"""

    def __init__(self, model_name: str = LLM_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model_name = model_name
        self.base_url = base_url

    def generate_answer(self, query: str, context: str, max_tokens: int = 300) -> str:
        prompt = f"""You will be given a question and a contextual text. 
        Answer the question based on the information you find in the context. Context:{context} Question: {query} Answer:"""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.3
                },
                timeout=READ_TIMEOUT
            )

            if response.status_code != 200:
                return f"Error: {response.text}"

            data = response.json()
            return data.get('response', 'No answer can be generated.').strip()
        except Exception as e:
            return f"LLM Error: {e}"