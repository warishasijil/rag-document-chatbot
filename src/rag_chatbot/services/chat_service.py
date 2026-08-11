from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

from rag_chatbot.config import Settings
from rag_chatbot.llm.llm_service import LLMService
from rag_chatbot.prompts.rag_prompt import (
    CONTEXTUALIZE_PROMPT,
    RAG_PROMPT,
)
from rag_chatbot.retrieval.retriever import DocumentRetriever


@dataclass
class ChatResponse:
    """Response returned by the conversational RAG pipeline."""

    answer: str
    sources: list[str]
    retrieval_query: str


class ChatService:
    """Coordinates retrieval, prompt construction, and LLM generation."""

    def __init__(
        self,
        retriever: DocumentRetriever,
        llm_service: LLMService,
    ) -> None:
        self.retriever = retriever
        self.llm = llm_service.model

    def answer(
        self,
        question: str,
        chat_history: list[BaseMessage],
    ) -> ChatResponse:
        """Answer a user question using conversational RAG."""

        history = chat_history[
            -Settings.MAX_HISTORY_MESSAGES:
        ]

        # 1. Create a standalone retrieval query

        retrieval_query = self._create_retrieval_query(
            question=question,
            chat_history=history,
        )

        # 2. Retrieve relevant documents from ChromaDB

        documents = self.retriever.retrieve(
            retrieval_query
        )

        
        # 3. Convert retrieved Documents into prompt context

        context = self._format_context(
            documents
        )

        # 4. Build the final RAG prompt

        prompt = RAG_PROMPT.invoke(
            {
                "question": retrieval_query,
                "context": context,
                "chat_history": history,
            }
        )

        # 5. Ask the LLM to generate the grounded answer

        response = self.llm.invoke(
            prompt.messages
        )

        
        # 6. Extract readable source references

        sources = self._extract_sources(
            documents
        )

        return ChatResponse(
            answer=str(response.content),
            sources=sources,
            retrieval_query=retrieval_query,
        )

    def _create_retrieval_query(
        self,
        question: str,
        chat_history: list[BaseMessage],
    ) -> str:
        """
        Rewrite conversational follow-up questions into
        standalone retrieval queries.
        """

        # No previous conversation means no rewriting is needed.
        if not chat_history:
            return question

        prompt = CONTEXTUALIZE_PROMPT.invoke(
            {
                "question": question,
                "chat_history": chat_history,
            }
        )

        response = self.llm.invoke(
            prompt.messages
        )

        return str(response.content).strip()

    @staticmethod
    def _format_context(
        documents: list[Document],
    ) -> str:
        """Format retrieved documents for the RAG prompt."""

        if not documents:
            return "No relevant documents were retrieved."

        formatted_documents = []

        for index, document in enumerate(
            documents,
            start=1,
        ):
            metadata = document.metadata

            location_parts = [
                metadata.get(
                    "file_name",
                    "Unknown source",
                )
            ]

            if "page" in metadata:
                location_parts.append(
                    f"page {metadata['page']}"
                )

            if "sheet" in metadata:
                location_parts.append(
                    f"sheet {metadata['sheet']}"
                )

            if "row" in metadata:
                location_parts.append(
                    f"row {metadata['row']}"
                )

            location = ", ".join(
                location_parts
            )

            formatted_documents.append(
                f"[Source {index} | {location}]\n"
                f"{document.page_content}"
            )

        return "\n\n".join(
            formatted_documents
        )

    @staticmethod
    def _extract_sources(
        documents: list[Document],
    ) -> list[str]:
        """Create human-readable references for retrieved sources."""

        sources = []

        for document in documents:
            metadata = document.metadata

            source_parts = [
                metadata.get(
                    "file_name",
                    "Unknown source",
                )
            ]

            if "page" in metadata:
                source_parts.append(
                    f"page {metadata['page']}"
                )

            if "sheet" in metadata:
                source_parts.append(
                    f"sheet {metadata['sheet']}"
                )

            if "row" in metadata:
                source_parts.append(
                    f"row {metadata['row']}"
                )

            source = " — ".join(
                source_parts
            )

            if source not in sources:
                sources.append(source)

        return sources