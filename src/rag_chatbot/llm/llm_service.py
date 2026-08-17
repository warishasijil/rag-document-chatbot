import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from rag_chatbot.config import Settings


class LLMService:
    """Create and expose the chat model used by the RAG pipeline."""

    def __init__(self):
        load_dotenv()

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY was not found. Add it to your .env file."
            )

        self._model = ChatGroq(
            api_key=api_key,
            model=Settings.LLM_MODEL_NAME,
            temperature=Settings.LLM_TEMPERATURE,
            max_retries=2,
        )

    @property
    def model(self) -> ChatGroq:
        return self._model