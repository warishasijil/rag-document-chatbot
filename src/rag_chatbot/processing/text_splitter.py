from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from rag_chatbot.config import Settings


class DocumentTextSplitter:
    """Splits documents into chunks for vector retrieval."""

    def __init__(
        self,
        chunk_size: int = Settings.CHUNK_SIZE,
        chunk_overlap: int = Settings.CHUNK_OVERLAP,
    ) -> None:
        self.text_splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                add_start_index=True,
            )
        )

    def split(
        self,
        documents: list[Document],
    ) -> list[Document]:
        """Split documents while preserving metadata."""

        if not documents:
            return []

        return self.text_splitter.split_documents(
            documents
        )