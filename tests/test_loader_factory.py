from pathlib import Path

import pytest

from rag_chatbot.ingestion.docx_loader import DOCXDocumentLoader
from rag_chatbot.ingestion.excel_loader import ExcelDocumentLoader
from rag_chatbot.ingestion.loader_factory import DocumentLoaderFactory
from rag_chatbot.ingestion.pdf_loader import PDFDocumentLoader
from rag_chatbot.ingestion.text_loader import TextDocumentLoader


@pytest.mark.parametrize(
    ("file_name", "expected_loader"),
    [
        (
            "document.txt",
            TextDocumentLoader,
        ),
        (
            "document.pdf",
            PDFDocumentLoader,
        ),
        (
            "document.docx",
            DOCXDocumentLoader,
        ),
        (
            "document.xlsx",
            ExcelDocumentLoader,
        ),
    ],
)
def test_factory_selects_correct_loader(
    file_name,
    expected_loader,
):
    """Factory should select loader based on file extension."""

    loader = DocumentLoaderFactory.create_loader(
        Path(file_name)
    )

    assert isinstance(
        loader,
        expected_loader,
    )


def test_factory_rejects_unsupported_format():
    """Unsupported extensions should raise a clear error."""

    with pytest.raises(
        ValueError,
        match="Unsupported file type",
    ):
        DocumentLoaderFactory.create_loader(
            Path("document.csv")
        )