from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

from rag_chatbot.config import Settings
from rag_chatbot.llm.llm_service import LLMService
from rag_chatbot.prompts.rag_prompt import CONTEXTUALIZE_PROMPT, RAG_PROMPT
from rag_chatbot.retrieval.retriever import DocumentRetriever


@dataclass
class ChatResponse:
    answer: str
    sources: list[str]
    retrieval_query: str


class ChatService:
    """Handles conversational retrieval and answer generation."""

    def __init__(self, retriever: DocumentRetriever, llm_service: LLMService):
        self.retriever = retriever
        self.llm = llm_service.model

    def answer(
        self,
        question: str,
        chat_history: list[BaseMessage],
    ) -> ChatResponse:
        history = chat_history[-Settings.MAX_HISTORY_MESSAGES:]

        retrieval_query = self._create_retrieval_query(question, history)
        documents = self.retriever.retrieve(retrieval_query)
        context = self._format_context(documents)

        prompt = RAG_PROMPT.invoke(
            {
                "question": retrieval_query,
                "context": context,
                "chat_history": history,
            }
        )

        response = self.llm.invoke(prompt.messages)

        return ChatResponse(
            answer=str(response.content),
            sources=self._extract_sources(documents),
            retrieval_query=retrieval_query,
        )

    def _create_retrieval_query(
        self,
        question: str,
        history: list[BaseMessage],
    ) -> str:
        """Turn a follow-up question into a standalone search query."""
        if not history:
            return question

        prompt = CONTEXTUALIZE_PROMPT.invoke(
            {
                "question": question,
                "chat_history": history,
            }
        )

        response = self.llm.invoke(prompt.messages)
        query = str(response.content).strip()

        return self._remove_duplicate_query(query)

    @staticmethod
    def _remove_duplicate_query(query: str) -> str:
        """Remove an accidental repeated query from LLM output."""
        midpoint = len(query) // 2

        if len(query) % 2 == 0 and query[:midpoint] == query[midpoint:]:
            return query[:midpoint].strip()

        return query

    @staticmethod
    def _format_context(documents: list[Document]) -> str:
        """Format retrieved documents for the RAG prompt."""
        sections = []

        for index, document in enumerate(documents, start=1):
            metadata = document.metadata
            source = metadata.get("file_name", "Unknown source")

            details = [source]

            if metadata.get("page") is not None:
                details.append(f"page {metadata['page']}")

            if metadata.get("sheet"):
                details.append(f"sheet {metadata['sheet']}")

            if metadata.get("row") is not None:
                details.append(f"row {metadata['row']}")

            label = ", ".join(details)

            sections.append(
                f"[Source {index} | {label}]\n"
                f"{document.page_content}"
            )

        return "\n\n".join(sections)

    @staticmethod
    def _extract_sources(documents: list[Document]) -> list[str]:
        """Return a readable, deduplicated list of retrieved sources."""
        sources = []

        for document in documents:
            metadata = document.metadata
            source = metadata.get("file_name", "Unknown source")

            if metadata.get("page") is not None:
                source += f" — page {metadata['page']}"

            if metadata.get("sheet"):
                source += f" — sheet {metadata['sheet']}"

            if metadata.get("row") is not None:
                source += f" — row {metadata['row']}"

            if source not in sources:
                sources.append(source)

        return sources