import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from rag_chatbot.config import Settings
from rag_chatbot.embeddings.embedding_service import EmbeddingService
from rag_chatbot.llm.llm_service import LLMService
from rag_chatbot.retrieval.retriever import DocumentRetriever
from rag_chatbot.services.chat_service import ChatService
from rag_chatbot.vectorstore.chroma_store import ChromaVectorStore


st.set_page_config(
    page_title="NexaTech RAG Assistant",
    page_icon="💬",
    layout="centered",
)


@st.cache_resource
def get_chat_service() -> ChatService:
    """Create the RAG services once and reuse them across Streamlit reruns."""
    embeddings = EmbeddingService()

    vector_store = ChromaVectorStore(
        embedding_function=embeddings.embeddings
    )

    retriever = DocumentRetriever(vector_store)
    llm_service = LLMService()

    return ChatService(retriever, llm_service)


def build_chat_history(messages: list[dict]) -> list:
    """Convert Streamlit chat messages into LangChain messages."""
    history = []

    for message in messages:
        if message["role"] == "user":
            history.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            history.append(AIMessage(content=message["content"]))

    return history


def show_sources(sources: list[str]) -> None:
    if not sources:
        return

    with st.expander("Sources"):
        for source in sources:
            st.markdown(f"- {source}")


def show_retrieval_query(query: str) -> None:
    if query:
        with st.expander("Retrieval details"):
            st.code(query, language=None)


def display_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            show_sources(message.get("sources", []))
            show_retrieval_query(message.get("retrieval_query", ""))


st.title("NexaTech RAG Assistant")
st.caption(
    "Ask questions about NexaTech policies, employee information, "
    "products and service centres."
)


with st.sidebar:
    st.header("About")

    st.write(
        "This chatbot uses retrieval-augmented generation to answer "
        "questions from the indexed NexaTech documents."
    )

    st.divider()

    st.subheader("Configuration")
    st.write(f"Embedding model: `{Settings.EMBEDDING_MODEL_NAME}`")
    st.write(f"LLM: `{Settings.LLM_MODEL_NAME}`")
    st.write(f"Retrieved chunks: `{Settings.RETRIEVAL_K}`")

    st.divider()

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = []


if not Settings.CHROMA_DIR.exists():
    st.error(
        "The vector database has not been built yet. "
        "Run `python scripts/build_index.py` first."
    )
    st.stop()


try:
    chat_service = get_chat_service()
except Exception as exc:
    st.error(f"Could not initialize the RAG system: {exc}")
    st.stop()


for message in st.session_state.messages:
    display_message(message)


question = st.chat_input("Ask a question about NexaTech...")


if question:
    # Build the history before adding the new question.
    history = build_chat_history(st.session_state.messages)

    user_message = {
        "role": "user",
        "content": question,
    }

    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the documents..."):
            try:
                response = chat_service.answer(
                    question=question,
                    chat_history=history,
                )

                answer = response.answer

                # Retrieved chunks are not useful citations when the system
                # explicitly says the answer was not found.
                not_found = (
                    "I couldn't find that information in the indexed documents."
                )

                sources = [] if answer.strip() == not_found else response.sources

                st.markdown(answer)
                show_sources(sources)
                show_retrieval_query(response.retrieval_query)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "retrieval_query": response.retrieval_query,
                    }
                )

            except Exception as exc:
                st.error(f"Something went wrong while answering: {exc}")