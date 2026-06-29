
import os
import fitz                       
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss                         
from rank_bm25 import BM25Okapi     
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, DOCUMENTS_DIR



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

        # fitz.open() reads the PDF
        doc = fitz.open(filepath)

        for page_number, page in enumerate(doc, start=1):
            text = page.get_text()   # extract plain text from this page

            # skip pages with almost no text
            if len(text.strip()) < 50:
                continue

            all_pages.append({
                "text":     text,
                "filename": filename,
                "page":     page_number
            })

        doc.close()

    print(f"[INFO] Loaded {len(all_pages)} pages from {len(pdf_files)} PDF(s).")
    return all_pages




def split_into_chunks(all_pages, chunk_size, chunk_overlap):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    all_chunks = []

    for page in all_pages:
        # split_text() returns a list of strings
        chunks = splitter.split_text(page["text"])

        for chunk_text in chunks:
            all_chunks.append({
                "text":     chunk_text,
                "filename": page["filename"],
                "page":     page["page"]
            })

    print(f"[INFO] Created {len(all_chunks)} chunks total.")
    return all_chunks


def build_faiss_index(all_chunks, model):

    print("[INFO] Generating embeddings (this may take a minute)...")

    # Extract just the text from each chunk
    texts = [chunk["text"] for chunk in all_chunks]


    embeddings = model.encode(texts, show_progress_bar=True)

    embeddings = np.array(embeddings, dtype="float32")

    # IndexFlatL2 does exact nearest-neighbour search using L2 (Euclidean) distance
    # embeddings.shape[1] is the vector dimension (384 for all-MiniLM-L6-v2)
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)

    # Add all vectors to the FAISS index
    faiss_index.add(embeddings)

    print(f"[INFO] FAISS index built with {faiss_index.ntotal} vectors.")
    return faiss_index, embeddings


def build_bm25_index(all_chunks):

    print("[INFO] Building BM25 index...")

    # Tokenize: split each chunk into individual words (lowercase)
    tokenized_corpus = [chunk["text"].lower().split() for chunk in all_chunks]

    bm25 = BM25Okapi(tokenized_corpus)

    print("[INFO] BM25 index built.")
    return bm25, tokenized_corpus


def retrieve(query, all_chunks, faiss_index, bm25, embedding_model, top_k=5):

    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding, dtype="float32")

    distances, indices = faiss_index.search(query_embedding, top_k)

    faiss_results = []
    for distance, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue

        score = float(1 / (1 + distance))

        faiss_results.append({
            "text":     all_chunks[idx]["text"],
            "score":    round(score, 4),
            "filename": all_chunks[idx]["filename"],
            "page":     all_chunks[idx]["page"],
            "source":   "faiss"     # just for debugging; not in final output
        })


    query_tokens = query.lower().split()

    bm25_scores = bm25.get_scores(query_tokens)

    # Get the indices of the top_k highest BM25 scores
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]

    # Normalize BM25 scores to a 0–1 range so they're comparable to FAISS scores
    max_bm25_score = bm25_scores[top_bm25_indices[0]] if bm25_scores[top_bm25_indices[0]] > 0 else 1

    bm25_results = []
    for idx in top_bm25_indices:
        raw_score = bm25_scores[idx]

        # Skip chunks with zero score (no keyword match at all)
        if raw_score <= 0:
            continue

        normalized_score = float(raw_score / max_bm25_score)

        bm25_results.append({
            "text":     all_chunks[idx]["text"],
            "score":    round(normalized_score, 4),
            "filename": all_chunks[idx]["filename"],
            "page":     all_chunks[idx]["page"],
            "source":   "bm25"
        })

    # ---- Merge, Deduplicate, and Sort ----

    # Combine both lists
    combined = faiss_results + bm25_results

    # Remove duplicates
    seen_texts = {}
    for result in combined:
        text = result["text"]
        if text not in seen_texts:
            seen_texts[text] = result
        else:
            # Keep whichever has the higher score
            if result["score"] > seen_texts[text]["score"]:
                seen_texts[text] = result

    # Convert back to a list and sort by score, highest first
    deduplicated = list(seen_texts.values())
    deduplicated.sort(key=lambda x: x["score"], reverse=True)

    # Take only top_k results
    top_results = deduplicated[:top_k]

    # Remove the "source" key — it was only used internally
    final_results = []
    for result in top_results:
        final_results.append({
            "text":     result["text"],
            "score":    result["score"],
            "filename": result["filename"],
            "page":     result["page"]
        })

    return final_results


def build_index():

    print(f"[INFO] Loading embedding model: {EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    # Run the pipeline
    all_pages  = load_pdfs(DOCUMENTS_DIR)
    all_chunks = split_into_chunks(all_pages, CHUNK_SIZE, CHUNK_OVERLAP)

    if not all_chunks:
        print("[ERROR] No chunks were created. Make sure the documents/ folder has PDFs.")
        return None, None, None, None

    faiss_index, _ = build_faiss_index(all_chunks, embedding_model)
    bm25, _         = build_bm25_index(all_chunks)

    print("[INFO] Index is ready. You can now call retrieve().")
    return all_chunks, faiss_index, bm25, embedding_model
