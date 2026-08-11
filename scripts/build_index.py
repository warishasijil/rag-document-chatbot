import shutil

from rag_chatbot.config import Settings
from rag_chatbot.embeddings.embedding_service import EmbeddingService
from rag_chatbot.ingestion.ingestion_service import (
    DocumentIngestionService,
)
from rag_chatbot.processing.text_splitter import (
    DocumentTextSplitter,
)
from rag_chatbot.vectorstore.chroma_store import (
    ChromaVectorStore,
)


def main() -> None:
    """Build the ChromaDB index from the raw documents."""

    print("\n================================")
    print("BUILDING VECTOR INDEX")
    print("================================")

    # --------------------------------------------------
    # 1. Load documents
    # --------------------------------------------------

    print("\n1. Loading documents...")

    ingestion_service = DocumentIngestionService()

    documents = ingestion_service.load_all()

    print(
        f"   Loaded {len(documents)} documents."
    )

    # --------------------------------------------------
    # 2. Split documents into chunks
    # --------------------------------------------------

    print("\n2. Splitting documents...")

    splitter = DocumentTextSplitter()

    chunks = splitter.split(documents)

    print(
        f"   Created {len(chunks)} chunks."
    )

    # --------------------------------------------------
    # 3. Rebuild ChromaDB from scratch
    # --------------------------------------------------

    print("\n3. Preparing ChromaDB...")

    if Settings.CHROMA_DIR.exists():
        shutil.rmtree(Settings.CHROMA_DIR)

        print(
            "   Removed existing vector database."
        )

    # --------------------------------------------------
    # 4. Load embedding model
    # --------------------------------------------------

    print("\n4. Loading embedding model...")

    embedding_service = EmbeddingService()

    print(
        f"   Model: "
        f"{Settings.EMBEDDING_MODEL_NAME}"
    )

    # --------------------------------------------------
    # 5. Create Chroma vector store
    # --------------------------------------------------

    print("\n5. Creating Chroma vector store...")

    vector_store = ChromaVectorStore(
        embedding_function=embedding_service.embeddings
    )

    # --------------------------------------------------
    # 6. Embed and index all chunks
    # --------------------------------------------------

    print("\n6. Embedding and indexing chunks...")

    document_ids = vector_store.add_documents(
        chunks
    )

    print(
        f"   Indexed {len(document_ids)} chunks."
    )

    print("\n================================")
    print("VECTOR INDEX COMPLETE")
    print("================================")

    print(
        f"\nDatabase location:\n"
        f"{Settings.CHROMA_DIR}"
    )


if __name__ == "__main__":
    main()