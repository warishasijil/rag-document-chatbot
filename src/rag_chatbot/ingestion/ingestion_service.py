from pathlib import Path

from langchain_core.documents import Document

from rag_chatbot.config import Settings
from rag_chatbot.ingestion.loader_factory import DocumentLoaderFactory


class DocumentIngestionService:
    """Load supported documents from the project data directory."""

    def __init__(self, data_directory: Path = Settings.RAW_DATA_DIR):
        self.data_directory = data_directory

    def load_file(self, file_path: Path) -> list[Document]:
        loader = DocumentLoaderFactory.create_loader(file_path)
        return loader.load(file_path)

    def load_all(self) -> list[Document]:
        if not self.data_directory.exists():
            raise FileNotFoundError(
                f"Data directory does not exist: {self.data_directory}"
            )

        documents = []

        for file_path in sorted(self.data_directory.iterdir()):
            if not file_path.is_file():
                continue

            if file_path.suffix.lower() not in Settings.SUPPORTED_EXTENSIONS:
                continue

            documents.extend(self.load_file(file_path))

        return documents