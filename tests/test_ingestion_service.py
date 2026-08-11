from pathlib import Path

import pytest

from rag_chatbot.ingestion.ingestion_service import (
    DocumentIngestionService,
)


def test_ingestion_service_loads_supported_files(
    tmp_path: Path,
):
    """
    Service should load supported files
    and ignore unsupported files.
    """

    # Create a supported TXT file
    txt_file = tmp_path / "policy.txt"

    txt_file.write_text(
        "Employees receive 25 days of annual leave.",
        encoding="utf-8",
    )

    # Create an unsupported CSV file
    unsupported_file = tmp_path / "notes.csv"

    unsupported_file.write_text(
        "column,value",
        encoding="utf-8",
    )

    # Point the ingestion service at the temporary directory
    service = DocumentIngestionService(
        data_directory=tmp_path
    )

    documents = service.load_all()

    # Only the supported TXT file should be loaded
    assert len(documents) == 1

    document = documents[0]

    assert (
        document.metadata["file_name"]
        == "policy.txt"
    )

    assert (
        document.metadata["file_type"]
        == ".txt"
    )

    assert (
        "25 days of annual leave"
        in document.page_content
    )


def test_ingestion_service_load_file(
    sample_txt: Path,
):
    """
    Service should be able to load
    one supported file directly.
    """

    service = DocumentIngestionService()

    documents = service.load_file(
        sample_txt
    )

    assert len(documents) == 1

    document = documents[0]

    assert (
        document.metadata["file_name"]
        == "sample.txt"
    )

    assert (
        "25 days of annual leave"
        in document.page_content
    )


def test_ingestion_service_missing_directory(
    tmp_path: Path,
):
    """
    Missing data directories should
    raise FileNotFoundError.
    """

    missing_directory = (
        tmp_path
        / "does_not_exist"
    )

    service = DocumentIngestionService(
        data_directory=missing_directory
    )

    with pytest.raises(
        FileNotFoundError
    ):
        service.load_all()


def test_ingestion_service_empty_directory(
    tmp_path: Path,
):
    """
    An existing empty directory should
    return an empty document list.
    """

    service = DocumentIngestionService(
        data_directory=tmp_path
    )

    documents = service.load_all()

    assert documents == []