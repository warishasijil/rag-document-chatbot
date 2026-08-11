from langchain_core.documents import Document

from rag_chatbot.vectorstore.chroma_store import ChromaVectorStore


class DocumentRetriever:
    """Retrieves relevant document chunks from ChromaDB."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
    ) -> None:
        """
        Initialize the retriever.

        Args:
            vector_store:
                The application's Chroma vector store.
        """
        self._retriever = vector_store.as_retriever()

    def retrieve(
        self,
        query: str,
    ) -> list[Document]:
        """
        Retrieve relevant documents for a query.

        Args:
            query:
                User query or rewritten standalone query.

        Returns:
            Relevant LangChain Document objects.
        """

        if not query.strip():
            return []

        return self._retriever.invoke(query)