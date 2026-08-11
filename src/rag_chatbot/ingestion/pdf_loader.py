from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

from rag_chatbot.ingestion.base_loader import BaseDocumentLoader


class PDFDocumentLoader(BaseDocumentLoader):
    """Loader for PDF files."""

    def load(self, file_path: Path) -> list[Document]:
        reader = PdfReader(file_path)

        documents: list[Document] = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()

            if not text or not text.strip():
                continue

            document = Document(
                page_content=text,
                metadata={
                    "source": str(file_path),
                    "file_name": file_path.name,
                    "file_type": file_path.suffix.lower(),
                    "page": page_number,
                },
            )

            documents.append(document)

        return documents