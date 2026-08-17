from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

from rag_chatbot.ingestion.base_loader import BaseDocumentLoader


class PDFDocumentLoader(BaseDocumentLoader):
    """Load text from PDF files page by page."""

    def load(self, file_path: Path) -> list[Document]:
        reader = PdfReader(file_path)
        documents = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()

            if not text:
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(file_path),
                        "file_name": file_path.name,
                        "file_type": file_path.suffix.lower(),
                        "page": page_number,
                    },
                )
            )

        return documents