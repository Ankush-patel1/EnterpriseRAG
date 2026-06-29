# config.py
# ---------------------------------------------------------------------------
# All project-wide constants live here.
# If you want to tweak chunk size, model name, etc. — change it here only.
# ---------------------------------------------------------------------------

# Folder that contains all the PDF documents to be indexed
DOCUMENTS_DIR = "documents"

# How many characters each text chunk should contain
CHUNK_SIZE = 800

# How many characters the next chunk should overlap with the previous one
# (so that sentences at chunk boundaries are not lost)
CHUNK_OVERLAP = 150

# How many results to return from retrieve()
TOP_K = 5

# Sentence-transformers model used to create vector embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
