from pathlib import Path


class Settings:
    """Central configuration for the RAG chatbot."""

    # --------------------------------------------------
    # Project directories
    # --------------------------------------------------

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"

    CHROMA_DIR = PROJECT_ROOT / "chroma_db"

    # --------------------------------------------------
    # Supported document formats
    # --------------------------------------------------

    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".docx",
        ".txt",
        ".xlsx",
    }

    # --------------------------------------------------
    # Text splitting
    # --------------------------------------------------

    CHUNK_SIZE = 300
    CHUNK_OVERLAP = 50

    # --------------------------------------------------
    # Embeddings
    # --------------------------------------------------

    EMBEDDING_MODEL_NAME = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # --------------------------------------------------
    # ChromaDB
    # --------------------------------------------------

    CHROMA_COLLECTION_NAME = "nexatech_documents"

    # --------------------------------------------------
    # Retrieval
    # --------------------------------------------------

    RETRIEVAL_K = 4

    # --------------------------------------------------
    # LLM
    # --------------------------------------------------

    LLM_MODEL_NAME = "llama-3.1-8b-instant"
    LLM_TEMPERATURE = 0.0

    # Number of previous messages passed to the model
    MAX_HISTORY_MESSAGES = 8