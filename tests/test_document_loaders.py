from rag_chatbot.ingestion.docx_loader import DOCXDocumentLoader
from rag_chatbot.ingestion.excel_loader import ExcelDocumentLoader
from rag_chatbot.ingestion.pdf_loader import PDFDocumentLoader
from rag_chatbot.ingestion.text_loader import TextDocumentLoader


def test_text_loader(sample_txt):
    """TXT loader should return text with useful metadata."""

    loader = TextDocumentLoader()

    documents = loader.load(
        sample_txt
    )

    assert len(documents) == 1

    document = documents[0]

    assert (
        "25 days of annual leave"
        in document.page_content
    )

    assert (
        document.metadata["file_name"]
        == "sample.txt"
    )

    assert (
        document.metadata["file_type"]
        == ".txt"
    )


def test_pdf_loader(sample_pdf):
    """PDF loader should preserve page metadata."""

    loader = PDFDocumentLoader()

    documents = loader.load(
        sample_pdf
    )

    assert len(documents) == 1

    document = documents[0]

    assert (
        "25 days of annual leave"
        in document.page_content
    )

    assert (
        document.metadata["file_name"]
        == "sample.pdf"
    )

    assert (
        document.metadata["page"]
        == 1
    )


def test_docx_loader(sample_docx):
    """DOCX loader should extract paragraph text."""

    loader = DOCXDocumentLoader()

    documents = loader.load(
        sample_docx
    )

    assert len(documents) == 1

    document = documents[0]

    assert (
        "Remote Working Policy"
        in document.page_content
    )

    assert (
        "three days per week"
        in document.page_content
    )

    assert (
        document.metadata["file_type"]
        == ".docx"
    )


def test_excel_loader(sample_xlsx):
    """Excel loader should create one Document per data row."""

    loader = ExcelDocumentLoader()

    documents = loader.load(
        sample_xlsx
    )

    assert len(documents) == 1

    document = documents[0]

    assert (
        "NexaBook Pro"
        in document.page_content
    )

    assert (
        "1299"
        in document.page_content
    )

    assert (
        document.metadata["sheet"]
        == "Products"
    )

    assert (
        document.metadata["row"]
        == 2
    )