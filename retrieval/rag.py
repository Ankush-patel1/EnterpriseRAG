import os
import fitz
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from rank_bm25 import BM25Okapi
from langchain.text_splitter import RecursiveCharacterTextSplitter

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


_state = {
    "all_chunks": None,
    "faiss_index": None,
    "bm25": None,
    "embed_model": None,
    "reranker": None,
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
    #splitting at paragraph and sentence boundaries to keep chunks coherent
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


def build_faiss_index(all_chunks, embed_model):
    print("[INFO] Generating embeddings (this may take a minute)...")
    texts = [chunk["text"] for chunk in all_chunks]
    embeddings = embed_model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")

    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)

    print(f"[INFO] FAISS index built with {faiss_index.ntotal} vectors.")
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
    Higher k = less emphasis on top ranks. k=60 is the standard default.
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
    More accurate than bi-encoder similarity but slower; applied to a small candidate set.
    """
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
    print(f"[INFO] Loading embedding model: {EMBEDDING_MODEL}")
    _state["embed_model"] = SentenceTransformer(EMBEDDING_MODEL)

    print(f"[INFO] Loading reranker model: {RERANKER_MODEL}")
    _state["reranker"] = CrossEncoder(RERANKER_MODEL)

    all_pages = load_pdfs(DOCUMENTS_DIR)
    all_chunks = split_into_chunks(all_pages, CHUNK_SIZE, CHUNK_OVERLAP)

    if not all_chunks:
        print("[ERROR] No chunks created. Make sure the documents/ folder has PDFs.")
        return False

    _state["all_chunks"] = all_chunks
    _state["faiss_index"] = build_faiss_index(all_chunks, _state["embed_model"])
    _state["bm25"] = build_bm25_index(all_chunks)

    print("[INFO] Index ready.")
    return True


def retrieve(query, top_k=TOP_K):
    """
    Main retrieval function.
    Pipeline: dense (FAISS) + sparse (BM25) → RRF merge → cross-encoder reranking.
    Lazy-initializes the index on the first call.
    """
    if _state["all_chunks"] is None:
        success = _build_index()
        if not success:
            return []

    all_chunks = _state["all_chunks"]

    # --- Dense retrieval (FAISS) ---
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

    # --- Cross-encoder reranking ---
    candidates = merged[:TOP_K_CANDIDATES]
    reranked = rerank(query, candidates, _state["reranker"])

    return reranked[:top_k]


def build_index():
    """Kept for backward compatibility with retrieval/main.py."""
    success = _build_index()
    if success:
        return _state["all_chunks"], _state["faiss_index"], _state["bm25"], _state["embed_model"]
    return None, None, None, None
