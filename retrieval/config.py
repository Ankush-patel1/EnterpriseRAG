import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
