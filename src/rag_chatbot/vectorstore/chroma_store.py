from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_chatbot.config import Settings


class ChromaVectorStore:
    """Small wrapper around the project's persistent Chroma database."""

    def __init__(
        self,
        embedding_function: Embeddings,
        persist_directory: Path = Settings.CHROMA_DIR,
        collection_name: str = Settings.CHROMA_COLLECTION_NAME,
    ):
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=str(persist_directory),
        )

    def add_documents(self, documents: list[Document]) -> list[str]:
        if not documents:
            return []

        return self.vector_store.add_documents(documents)

    def similarity_search(
        self,
        query: str,
        k: int = Settings.RETRIEVAL_K,
    ) -> list[Document]:
        return self.vector_store.similarity_search(query, k=k)

    def as_retriever(self):
        return self.vector_store.as_retriever(
            search_kwargs={"k": Settings.RETRIEVAL_K}
        )