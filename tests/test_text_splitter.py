from langchain_core.documents import Document

from rag_chatbot.processing.text_splitter import (
    DocumentTextSplitter,
)


def test_text_splitter_creates_multiple_chunks():
    """Long documents should be split into multiple chunks."""

    text = (
        "Annual leave policy information. "
        * 100
    )

    document = Document(
        page_content=text,
        metadata={
            "file_name": "handbook.pdf",
            "page": 1,
        },
    )

    splitter = DocumentTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
    )

    chunks = splitter.split(
        [document]
    )

    assert len(chunks) > 1


def test_text_splitter_preserves_metadata():
    """Chunking should preserve source metadata."""

    document = Document(
        page_content=(
            "Remote working information. "
            * 50
        ),
        metadata={
            "file_name": "policy.docx",
            "department": "HR",
        },
    )

    splitter = DocumentTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
    )

    chunks = splitter.split(
        [document]
    )

    for chunk in chunks:

        assert (
            chunk.metadata["file_name"]
            == "policy.docx"
        )

        assert (
            chunk.metadata["department"]
            == "HR"
        )

        assert (
            "start_index"
            in chunk.metadata
        )


def test_empty_document_list_returns_empty_list():
    """Splitting no documents should return no chunks."""

    splitter = DocumentTextSplitter()

    chunks = splitter.split([])

    assert chunks == []