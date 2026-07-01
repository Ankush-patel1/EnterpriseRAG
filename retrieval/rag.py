import os
import fitz
import numpy as np
import faiss
import threading
from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from retrieval.config import (
        CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL,
        DOCUMENTS_DIR, TOP_K, TOP_K_CANDIDATES, RERANKER_MODEL
    )
except ImportError:
    from config import (
        CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL,
        DOCUMENTS_DIR, TOP_K, TOP_K_CANDIDATES, RERANKER_MODEL
    )

# Lock to prevent race conditions / concurrent builds during lazy initialization
_index_lock = threading.Lock()

_state = {
    "all_chunks": None,
    "faiss_index": None,
    "bm25": None,
    "use_gemini_embeddings": False,
    "embed_model": None,  # SentenceTransformer (if local)
    "reranker": None,     # CrossEncoder (if local)
}


def load_pdfs(documents_dir):
    all_pages = []

    if not os.path.exists(documents_dir):
        print(f"[ERROR] Folder '{documents_dir}' does not exist.")
        return all_pages

    pdf_files = [f for f in os.listdir(documents_dir) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"[WARNING] No PDF files found in '{documents_dir}'.")
        return all_pages

    for filename in pdf_files:
        filepath = os.path.join(documents_dir, filename)
        print(f"[INFO] Loading: {filename}")

        doc = fitz.open(filepath)

        for page_number, page in enumerate(doc, start=1):
            # Extract text as blocks to preserve document structure (headings, paragraphs)
            blocks = page.get_text("blocks")
            structured_text = "\n\n".join(
                block[4].strip() for block in blocks if block[4].strip()
            )

            if len(structured_text.strip()) < 50:
                continue

            all_pages.append({
                "text": structured_text,
                "filename": filename,
                "page": page_number
            })

        doc.close()

    print(f"[INFO] Loaded {len(all_pages)} pages from {len(pdf_files)} PDF(s).")
    return all_pages


def split_into_chunks(all_pages, chunk_size, chunk_overlap):
    # splitting at paragraph and sentence boundaries to keep chunks coherent
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "]
    )

    all_chunks = []

    for page in all_pages:
        chunks = splitter.split_text(page["text"])

        for chunk_text in chunks:
            all_chunks.append({
                "text": chunk_text,
                "filename": page["filename"],
                "page": page["page"]
            })

    print(f"[INFO] Created {len(all_chunks)} chunks total.")
    return all_chunks


def get_gemini_embeddings(texts):
    """Fetches embeddings from the Google GenAI API to save memory in production."""
    from google import genai
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    
    client = genai.Client(api_key=api_key)
    
    print(f"[INFO] Requesting Gemini embeddings for {len(texts)} chunks...")
    batch_size = 100
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client.models.embed_content(
            model="text-embedding-004",
            contents=batch
        )
        for emb in response.embeddings:
            all_embeddings.append(emb.values)
            
    return all_embeddings


def build_faiss_index(all_chunks, embed_model=None, use_gemini=False):
    print("[INFO] Generating embeddings...")
    texts = [chunk["text"] for chunk in all_chunks]
    
    if use_gemini:
        embeddings = get_gemini_embeddings(texts)
    else:
        # Lazy import SentenceTransformer to save RAM when using Gemini API
        from sentence_transformers import SentenceTransformer
        if embed_model is None:
            embed_model = SentenceTransformer(EMBEDDING_MODEL)
        embeddings = embed_model.encode(texts, show_progress_bar=True)
        
    embeddings = np.array(embeddings, dtype="float32")

    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)

    print(f"[INFO] FAISS index built with {faiss_index.ntotal} vectors (dimension={dimension}).")
    return faiss_index


def build_bm25_index(all_chunks):
    print("[INFO] Building BM25 index...")
    tokenized_corpus = [chunk["text"].lower().split() for chunk in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    print("[INFO] BM25 index built.")
    return bm25


def rrf_merge(faiss_results, bm25_results, k=60):
    """
    Reciprocal Rank Fusion — merges two ranked lists.
    Formula: score(d) = sum of 1 / (k + rank) across all lists that contain d.
    """
    rrf_scores = {}
    chunk_map = {}

    for rank, result in enumerate(faiss_results):
        key = result["text"]
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank + 1)
        chunk_map[key] = result

    for rank, result in enumerate(bm25_results):
        key = result["text"]
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank + 1)
        if key not in chunk_map:
            chunk_map[key] = result

    sorted_keys = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)

    merged = []
    for key in sorted_keys:
        chunk = dict(chunk_map[key])
        chunk["score"] = round(rrf_scores[key], 6)
        merged.append(chunk)

    return merged


def rerank(query, chunks, reranker_model):
    """
    Cross-encoder reranking — scores each (query, chunk) pair directly.
    If no reranker is present (e.g. low-memory environment), returns chunks sorted by RRF.
    """
    if reranker_model is None:
        return chunks
        
    pairs = [(query, chunk["text"]) for chunk in chunks]
    scores = reranker_model.predict(pairs)

    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)

    reranked = []
    for score, chunk in ranked:
        chunk = dict(chunk)
        chunk["score"] = round(float(score), 4)
        reranked.append(chunk)

    return reranked


def _build_index():
    """Loads models, parses PDFs, and builds FAISS + BM25 indexes into module state."""
    api_key = os.getenv("GEMINI_API_KEY")
    use_gemini = bool(api_key)
    
    embed_model = None
    reranker = None
    
    if use_gemini:
        print("[INFO] Low-memory mode: Using Gemini API (text-embedding-004) for embeddings.")
    else:
        print(f"[INFO] High-memory mode: Loading local embedding model: {EMBEDDING_MODEL}")
        from sentence_transformers import SentenceTransformer, CrossEncoder
        embed_model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"[INFO] Loading reranker model: {RERANKER_MODEL}")
        reranker = CrossEncoder(RERANKER_MODEL)

    all_pages = load_pdfs(DOCUMENTS_DIR)
    all_chunks = split_into_chunks(all_pages, CHUNK_SIZE, CHUNK_OVERLAP if 'CHUNK_OVERLAP' in globals() else 150)

    if not all_chunks:
        print("[ERROR] No chunks created. Make sure the documents/ folder has PDFs.")
        return False

    faiss_index = build_faiss_index(all_chunks, embed_model, use_gemini=use_gemini)
    bm25 = build_bm25_index(all_chunks)

    # Atomic assignment to global state at the end to prevent race conditions
    _state["embed_model"] = embed_model
    _state["reranker"] = reranker
    _state["all_chunks"] = all_chunks
    _state["faiss_index"] = faiss_index
    _state["bm25"] = bm25
    _state["use_gemini_embeddings"] = use_gemini

    print("[INFO] Index ready.")
    return True


def retrieve(query, top_k=TOP_K):
    """
    Main retrieval function.
    Pipeline: dense (FAISS) + sparse (BM25) → RRF merge → optional cross-encoder reranking.
    """
    if _state["all_chunks"] is None:
        with _index_lock:
            if _state["all_chunks"] is None:
                success = _build_index()
                if not success:
                    return []

    all_chunks = _state["all_chunks"]

    # --- Dense retrieval (FAISS) ---
    if _state["use_gemini_embeddings"]:
        from google import genai
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.embed_content(
            model="text-embedding-004",
            contents=query
        )
        query_embedding = [response.embeddings[0].values]
    else:
        query_embedding = _state["embed_model"].encode([query])
        
    query_embedding = np.array(query_embedding, dtype="float32")
    distances, indices = _state["faiss_index"].search(query_embedding, TOP_K_CANDIDATES)

    faiss_results = []
    for distance, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        faiss_results.append({
            "text": all_chunks[idx]["text"],
            "filename": all_chunks[idx]["filename"],
            "page": all_chunks[idx]["page"],
            "score": 0.0
        })

    # --- Sparse retrieval (BM25) ---
    query_tokens = query.lower().split()
    bm25_scores = _state["bm25"].get_scores(query_tokens)
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:TOP_K_CANDIDATES]

    bm25_results = []
    for idx in top_bm25_indices:
        if bm25_scores[idx] <= 0:
            continue
        bm25_results.append({
            "text": all_chunks[idx]["text"],
            "filename": all_chunks[idx]["filename"],
            "page": all_chunks[idx]["page"],
            "score": 0.0
        })

    # --- RRF merge ---
    merged = rrf_merge(faiss_results, bm25_results)

    if not merged:
        return []

    # --- Optional Cross-encoder reranking ---
    candidates = merged[:TOP_K_CANDIDATES]
    reranked = rerank(query, candidates, _state["reranker"])

    return reranked[:top_k]


def build_index():
    """Kept for backward compatibility."""
    success = _build_index()
    if success:
        return _state["all_chunks"], _state["faiss_index"], _state["bm25"], _state["embed_model"]
    return None, None, None, None
