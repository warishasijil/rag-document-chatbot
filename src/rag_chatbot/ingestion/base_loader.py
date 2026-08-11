from abc import ABC, abstractmethod
from pathlib import Path

from langchain_core.documents import Document


class BaseDocumentLoader(ABC):
    """Abstract base class for all document loaders."""

    @abstractmethod
    def load(self, file_path: Path) -> list[Document]:
        """
        Load a file and convert it into LangChain Documents.

        Args:
            file_path: Path to the source file.

        Returns:
            A list of LangChain Document objects.
        """
        raise NotImplementedError