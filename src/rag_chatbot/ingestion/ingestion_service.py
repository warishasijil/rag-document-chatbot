from pathlib import Path

from langchain_core.documents import Document

from rag_chatbot.config import Settings
from rag_chatbot.ingestion.loader_factory import DocumentLoaderFactory


class DocumentIngestionService:
    """Service responsible for ingesting supported documents."""

    def __init__(
        self,
        data_directory: Path | None = None,
    ) -> None:
        self.data_directory = (
            data_directory
            if data_directory is not None
            else Settings.RAW_DATA_DIR
        )

    def load_file(
        self,
        file_path: Path,
    ) -> list[Document]:
        """Load a single supported file."""

        loader = DocumentLoaderFactory.create_loader(
            file_path
        )

        return loader.load(file_path)

    def load_all(self) -> list[Document]:
        """Load all supported files from the data directory."""

        if not self.data_directory.exists():
            raise FileNotFoundError(
                f"Data directory does not exist: "
                f"{self.data_directory}"
            )

        documents: list[Document] = []

        for file_path in sorted(
            self.data_directory.iterdir()
        ):
            if not file_path.is_file():
                continue

            if (
                file_path.suffix.lower()
                not in Settings.SUPPORTED_EXTENSIONS
            ):
                continue

            loaded_documents = self.load_file(
                file_path
            )

            documents.extend(
                loaded_documents
            )

        return documents