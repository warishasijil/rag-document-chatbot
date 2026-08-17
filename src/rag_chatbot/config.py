from pathlib import Path


class Settings:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    CHROMA_DIR = PROJECT_ROOT / "chroma_db"

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".xlsx"}

    CHUNK_SIZE = 300
    CHUNK_OVERLAP = 50

    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    CHROMA_COLLECTION_NAME = "nexatech_documents"
    RETRIEVAL_K = 4

    LLM_MODEL_NAME = "openai/gpt-oss-20b"
    LLM_TEMPERATURE = 0.0

    MAX_HISTORY_MESSAGES = 8