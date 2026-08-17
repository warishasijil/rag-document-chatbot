import shutil

from rag_chatbot.config import Settings
from rag_chatbot.embeddings.embedding_service import EmbeddingService
from rag_chatbot.ingestion.ingestion_service import DocumentIngestionService
from rag_chatbot.processing.text_splitter import DocumentTextSplitter
from rag_chatbot.vectorstore.chroma_store import ChromaVectorStore


def main() -> None:
    print("Loading documents...")
    ingestion = DocumentIngestionService()
    documents = ingestion.load_all()
    print(f"Loaded {len(documents)} documents.")

    print("Splitting documents into chunks...")
    splitter = DocumentTextSplitter()
    chunks = splitter.split(documents)
    print(f"Created {len(chunks)} chunks.")

    # Rebuild from scratch so repeated runs do not duplicate documents.
    if Settings.CHROMA_DIR.exists():
        print("Removing existing Chroma index...")
        shutil.rmtree(Settings.CHROMA_DIR)

    print("Loading embedding model...")
    embeddings = EmbeddingService()

    print("Building Chroma index...")
    vector_store = ChromaVectorStore(
        embedding_function=embeddings.embeddings
    )
    vector_store.add_documents(chunks)

    print(f"\nIndex built successfully at: {Settings.CHROMA_DIR}")


if __name__ == "__main__":
    main()