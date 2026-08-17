from pathlib import Path

from rag_chatbot.ingestion.base_loader import BaseDocumentLoader
from rag_chatbot.ingestion.docx_loader import DOCXDocumentLoader
from rag_chatbot.ingestion.excel_loader import ExcelDocumentLoader
from rag_chatbot.ingestion.pdf_loader import PDFDocumentLoader
from rag_chatbot.ingestion.text_loader import TextDocumentLoader


class DocumentLoaderFactory:
    """Choose the correct loader for a file type."""

    _LOADERS = {
        ".txt": TextDocumentLoader,
        ".pdf": PDFDocumentLoader,
        ".docx": DOCXDocumentLoader,
        ".xlsx": ExcelDocumentLoader,
    }

    @classmethod
    def create_loader(cls, file_path: Path) -> BaseDocumentLoader:
        extension = file_path.suffix.lower()
        loader_class = cls._LOADERS.get(extension)

        if loader_class is None:
            raise ValueError(f"Unsupported file type: {extension}")

        return loader_class()