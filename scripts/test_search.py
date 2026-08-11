from rag_chatbot.config import Settings
from rag_chatbot.embeddings.embedding_service import (
    EmbeddingService,
)
from rag_chatbot.vectorstore.chroma_store import (
    ChromaVectorStore,
)


def main() -> None:
    """Test semantic search against the ChromaDB index."""

    print("\n================================")
    print("SEMANTIC SEARCH TEST")
    print("================================")

    # --------------------------------------------------
    # 1. Load the same embedding model
    # --------------------------------------------------

    embedding_service = EmbeddingService()

    # --------------------------------------------------
    # 2. Connect to the existing Chroma database
    # --------------------------------------------------

    vector_store = ChromaVectorStore(
        embedding_function=embedding_service.embeddings
    )

    # --------------------------------------------------
    # 3. Test queries
    # --------------------------------------------------

    queries = [
        (
            "How much holiday does a full-time "
            "employee get each year?"
        ),
        (
            "What do staff need to use when connecting "
            "to internal systems from home?"
        ),
        (
            "Which laptop comes with a three-year "
            "warranty?"
        ),
        (
            "Where can I get drop-off-only support?"
        ),
    ]

    for query in queries:

        print("\n================================")
        print(f"QUERY: {query}")
        print("================================")

        results = vector_store.similarity_search(
            query=query,
            k=3,
        )

        for rank, document in enumerate(
            results,
            start=1,
        ):
            print(f"\nResult #{rank}")

            print(
                f"Source: "
                f"{document.metadata.get('file_name')}"
            )

            if "page" in document.metadata:
                print(
                    f"Page: "
                    f"{document.metadata['page']}"
                )

            if "sheet" in document.metadata:
                print(
                    f"Sheet: "
                    f"{document.metadata['sheet']}"
                )

            if "row" in document.metadata:
                print(
                    f"Row: "
                    f"{document.metadata['row']}"
                )

            print("\nContent:")
            print(document.page_content)


if __name__ == "__main__":
    main()