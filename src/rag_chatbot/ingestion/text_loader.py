from pathlib import Path

from langchain_core.documents import Document

from rag_chatbot.ingestion.base_loader import BaseDocumentLoader


class TextDocumentLoader(BaseDocumentLoader):
    """Load plain text files."""

    def load(self, file_path: Path) -> list[Document]:
        text = file_path.read_text(encoding="utf-8").strip()

        if not text:
            return []

        return [
            Document(
                page_content=text,
                metadata={
                    "source": str(file_path),
                    "file_name": file_path.name,
                    "file_type": file_path.suffix.lower(),
                },
            )
        ]