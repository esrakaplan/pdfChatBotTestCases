# ================= CONFIGURATION =================
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "mistral:latest"
CHUNK_SIZE = 300        # Maximum characters per chunk
CHUNK_OVERLAP = 50      # Overlapping characters between chunks
TOP_K_RESULTS = 3       # Number of top chunks to retrieve
MODEL_CONN_TIMEOUT = 60
READ_TIMEOUT = 180
MAX_TOKEN = 300