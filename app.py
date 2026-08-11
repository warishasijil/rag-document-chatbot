import streamlit as st

from langchain_core.messages import AIMessage, HumanMessage

from rag_chatbot.config import Settings
from rag_chatbot.embeddings.embedding_service import EmbeddingService
from rag_chatbot.llm.llm_service import LLMService
from rag_chatbot.retrieval.retriever import DocumentRetriever
from rag_chatbot.services.chat_service import ChatService
from rag_chatbot.vectorstore.chroma_store import ChromaVectorStore


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NexaTech RAG Assistant",
    page_icon="📚",
    layout="centered",
)


# ============================================================
# APPLICATION SERVICES
# ============================================================

@st.cache_resource(show_spinner=False)
def get_chat_service() -> ChatService:
    """
    Create and cache the services required by the RAG application.

    Streamlit reruns the Python script whenever the user interacts
    with the page. Caching prevents expensive resources such as the
    embedding model from being recreated on every interaction.
    """

    embedding_service = EmbeddingService()

    vector_store = ChromaVectorStore(
        embedding_function=embedding_service.embeddings
    )

    retriever = DocumentRetriever(
        vector_store=vector_store
    )

    llm_service = LLMService()

    return ChatService(
        retriever=retriever,
        llm_service=llm_service,
    )


# ============================================================
# CHAT HISTORY CONVERSION
# ============================================================

def convert_to_langchain_history(
    messages: list[dict],
) -> list:
    """
    Convert Streamlit session messages into LangChain messages.

    LangChain models use HumanMessage and AIMessage objects for
    conversational context.
    """

    history = []

    for message in messages:

        role = message.get("role")
        content = message.get("content", "")

        if role == "user":

            history.append(
                HumanMessage(
                    content=content
                )
            )

        elif role == "assistant":

            history.append(
                AIMessage(
                    content=content
                )
            )

    return history


# ============================================================
# SOURCE DISPLAY
# ============================================================

def display_sources(
    sources: list[str],
) -> None:
    """Display source references underneath an assistant answer."""

    if not sources:
        return

    st.markdown("**Sources**")

    for source in sources:
        st.caption(f"• {source}")


# ============================================================
# PAGE HEADER
# ============================================================

st.title("📚 NexaTech Document Assistant")

st.caption(
    "Ask questions about the indexed company documents. "
    "Answers are generated using retrieval-augmented generation "
    "with LangChain, ChromaDB and Groq."
)


# ============================================================
# CHECK VECTOR DATABASE
# ============================================================

if not Settings.CHROMA_DIR.exists():

    st.error(
        "The ChromaDB vector index has not been built yet."
    )

    st.write(
        "Run the following command from the project root:"
    )

    st.code(
        "python scripts/build_index.py",
        language="powershell",
    )

    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("RAG Configuration")

    st.markdown(
        f"**LLM**  \n"
        f"`{Settings.LLM_MODEL_NAME}`"
    )

    st.markdown(
        f"**Embedding model**  \n"
        f"`{Settings.EMBEDDING_MODEL_NAME}`"
    )

    st.markdown(
        f"**Retrieved chunks**  \n"
        f"`{Settings.RETRIEVAL_K}`"
    )

    st.markdown(
        f"**Chunk size**  \n"
        f"`{Settings.CHUNK_SIZE}`"
    )

    st.markdown(
        f"**Chunk overlap**  \n"
        f"`{Settings.CHUNK_OVERLAP}`"
    )

    st.divider()

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# DISPLAY EXISTING CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    role = message["role"]

    with st.chat_message(role):

        st.markdown(
            message["content"]
        )

        # Show sources for previous assistant responses
        if (
            role == "assistant"
            and message.get("sources")
        ):

            display_sources(
                message["sources"]
            )

        # Keep retrieval information visible after reruns
        if (
            role == "assistant"
            and message.get("retrieval_query")
        ):

            with st.expander(
                "Retrieval details"
            ):

                st.markdown(
                    "**Query sent to ChromaDB:**"
                )

                st.code(
                    message["retrieval_query"]
                )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about the documents..."
)


# ============================================================
# PROCESS NEW QUESTION
# ============================================================

if question:

    # --------------------------------------------------------
    # Convert PREVIOUS messages to LangChain history
    #
    # Important:
    # We do this before adding the new question because
    # ChatService receives the current question separately.
    # --------------------------------------------------------

    chat_history = convert_to_langchain_history(
        st.session_state.messages
    )

    # --------------------------------------------------------
    # Save user message
    # --------------------------------------------------------

    user_message = {
        "role": "user",
        "content": question,
    }

    st.session_state.messages.append(
        user_message
    )

    # --------------------------------------------------------
    # Display user message immediately
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(question)

    # --------------------------------------------------------
    # Generate assistant response
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        try:

            with st.spinner(
                "Searching documents and generating answer..."
            ):

                chat_service = get_chat_service()

                response = chat_service.answer(
                    question=question,
                    chat_history=chat_history,
                )

            # ------------------------------------------------
            # Display answer
            # ------------------------------------------------

            st.markdown(
                response.answer
            )

            # ------------------------------------------------
            # Display retrieved sources
            # ------------------------------------------------

            display_sources(
                response.sources
            )

            # ------------------------------------------------
            # Display retrieval query
            # ------------------------------------------------

            with st.expander(
                "Retrieval details"
            ):

                st.markdown(
                    "**Query sent to ChromaDB:**"
                )

                st.code(
                    response.retrieval_query
                )

            # ------------------------------------------------
            # Save assistant response to visible history
            # ------------------------------------------------

            assistant_message = {
                "role": "assistant",
                "content": response.answer,
                "sources": response.sources,
                "retrieval_query": (
                    response.retrieval_query
                ),
            }

            st.session_state.messages.append(
                assistant_message
            )

        except Exception as error:

            st.error(
                "The chatbot could not generate a response."
            )

            st.exception(error)