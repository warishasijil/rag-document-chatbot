from pathlib import Path

from docx import Document as WordDocument
from langchain_core.documents import Document

from rag_chatbot.ingestion.base_loader import BaseDocumentLoader


class DOCXDocumentLoader(BaseDocumentLoader):
    """Loader for Microsoft Word (.docx) files."""

    def load(self, file_path: Path) -> list[Document]:
        word_document = WordDocument(file_path)

        paragraphs = [
            paragraph.text.strip()
            for paragraph in word_document.paragraphs
            if paragraph.text.strip()
        ]

        text = "\n\n".join(paragraphs)

        if not text:
            return []

        document = Document(
            page_content=text,
            metadata={
                "source": str(file_path),
                "file_name": file_path.name,
                "file_type": file_path.suffix.lower(),
            },
        )

        return [document]