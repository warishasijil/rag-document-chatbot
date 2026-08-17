from langchain_core.documents import Document

from rag_chatbot.vectorstore.chroma_store import ChromaVectorStore


class DocumentRetriever:
    """Retrieve relevant document chunks from the vector store."""

    def __init__(self, vector_store: ChromaVectorStore):
        self.retriever = vector_store.as_retriever()

    def retrieve(self, query: str) -> list[Document]:
        if not query.strip():
            return []

        return self.retriever.invoke(query)