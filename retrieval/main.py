try:
    from retrieval.rag import retrieve, build_index
    from retrieval.config import TOP_K
except ImportError:
    from rag import retrieve, build_index
    from config import TOP_K


def main():
    # Build the index explicitly (retrieve() does this lazily, but here we do it upfront)
    all_chunks, _, _, _ = build_index()

    if all_chunks is None:
        print("Could not build index. Please check the documents/ folder.")
        return

    queries = [
        "leave policy",
        "employee benefits",
        "performance review process",
    ]

    for query in queries:
        print("\n" + "=" * 60)
        print(f"Query: {query}")
        print("=" * 60)

        results = retrieve(query, top_k=TOP_K)

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
