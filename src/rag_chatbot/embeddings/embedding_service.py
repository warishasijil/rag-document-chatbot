from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from rag_chatbot.config import Settings


class EmbeddingService:
    """Load the embedding model used for indexing and retrieval."""

    def __init__(self, model_name: str = Settings.EMBEDDING_MODEL_NAME):
        self._embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )

    @property
    def embeddings(self) -> Embeddings:
        return self._embeddings