from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from rag_chatbot.config import Settings


class EmbeddingService:
    """Provides the embedding model used by the RAG system."""

    def __init__(
        self,
        model_name: str = Settings.EMBEDDING_MODEL_NAME,
    ) -> None:
        self._embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={
                "normalize_embeddings": True,
            },
        )

    @property
    def embeddings(self) -> Embeddings:
        """Return the configured LangChain embedding model."""
        return self._embeddings