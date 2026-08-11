import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from rag_chatbot.config import Settings


class LLMService:
    """Provides the Groq-hosted language model."""

    def __init__(self) -> None:
        # Load variables from the .env file
        load_dotenv()

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY was not found. "
                "Add it to your .env file."
            )

        self._model = ChatGroq(
            model=Settings.LLM_MODEL_NAME,
            temperature=Settings.LLM_TEMPERATURE,
            max_retries=2,
        )

    @property
    def model(self) -> ChatGroq:
        """Return the configured Groq chat model."""
        return self._model
    