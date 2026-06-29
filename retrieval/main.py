
from rag import build_index, retrieve
from config import TOP_K


def main():
    # Step 1: Build the index (loads PDFs, creates FAISS + BM25)
    all_chunks, faiss_index, bm25, embedding_model = build_index()

    if all_chunks is None:
        print("Could not build index. Please check the documents/ folder.")
        return

    # Step 2: Try a few example queries
    queries = [
        "leave policy",
        "employee benefits",
        "performance review process",
    ]

    for query in queries:
        print("\n" + "=" * 60)
        print(f"Query: {query}")
        print("=" * 60)

        results = retrieve(
            query=query,
            all_chunks=all_chunks,
            faiss_index=faiss_index,
            bm25=bm25,
            embedding_model=embedding_model,
            top_k=TOP_K
        )

        if not results:
            print("No results found.")
            continue

        for i, result in enumerate(results, start=1):
            print(f"\nResult #{i}")
            print(f"  File  : {result['filename']}")
            print(f"  Page  : {result['page']}")
            print(f"  Score : {result['score']}")
            print(f"  Text  : {result['text'][:200]}...")


if __name__ == "__main__":
    main()
