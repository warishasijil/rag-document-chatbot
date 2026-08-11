from langchain_core.messages import AIMessage, HumanMessage

from rag_chatbot.embeddings.embedding_service import EmbeddingService
from rag_chatbot.llm.llm_service import LLMService
from rag_chatbot.retrieval.retriever import DocumentRetriever
from rag_chatbot.services.chat_service import ChatService
from rag_chatbot.vectorstore.chroma_store import ChromaVectorStore


def print_response(question, response):
    """Print a RAG response in a readable format."""

    print("\n" + "=" * 70)
    print(f"QUESTION: {question}")
    print("=" * 70)

    print("\nRetrieval query:")
    print(response.retrieval_query)

    print("\nAnswer:")
    print(response.answer)

    print("\nSources:")

    if response.sources:
        for source in response.sources:
            print(f"- {source}")
    else:
        print("- No sources returned")


def main():
    """Test the complete conversational RAG pipeline."""

    print("\n======================================")
    print("INITIALIZING RAG SYSTEM")
    print("======================================")

    # 1. Load embedding model
    

    print("\nLoading embedding model...")

    embedding_service = EmbeddingService()

    # 2. Connect to existing ChromaDB

    print("Connecting to ChromaDB...")

    vector_store = ChromaVectorStore(
        embedding_function=embedding_service.embeddings
    )

    
    # 3. Create retriever

    print("Creating retriever...")

    retriever = DocumentRetriever(
        vector_store=vector_store
    )

    # 4. Connect to Groq LLM

    print("Connecting to Groq...")

    llm_service = LLMService()

    
    # 5. Create conversational RAG service

    chat_service = ChatService(
        retriever=retriever,
        llm_service=llm_service,
    )

    print("\nRAG system initialized successfully.")

    # TEST 1: Simple document question

    chat_history = []

    question_1 = (
        "How many days of annual leave do "
        "full-time employees receive?"
    )

    response_1 = chat_service.answer(
        question=question_1,
        chat_history=chat_history,
    )

    print_response(
        question_1,
        response_1,
    )

    # Add first conversation turn to history
    chat_history.append(
        HumanMessage(content=question_1)
    )

    chat_history.append(
        AIMessage(content=response_1.answer)
    )

    # TEST 2: Conversational follow-up

    question_2 = (
        "How many can they carry forward?"
    )

    response_2 = chat_service.answer(
        question=question_2,
        chat_history=chat_history,
    )

    print_response(
        question_2,
        response_2,
    )

    # TEST 3: Excel retrieval
    

    question_3 = (
        "Which laptop has a three-year warranty?"
    )

    response_3 = chat_service.answer(
        question=question_3,
        chat_history=[],
    )

    print_response(
        question_3,
        response_3,
    )

    # Create a separate conversation about the laptop
    laptop_history = [
        HumanMessage(content=question_3),
        AIMessage(content=response_3.answer),
    ]

    
    # TEST 4: Follow-up referring to previous product
    

    question_4 = "How much does it cost?"

    response_4 = chat_service.answer(
        question=question_4,
        chat_history=laptop_history,
    )

    print_response(
        question_4,
        response_4,
    )

    # TEST 5: Hallucination / missing information
    

    question_5 = "Who is the CEO of NexaTech?"

    response_5 = chat_service.answer(
        question=question_5,
        chat_history=[],
    )

    print_response(
        question_5,
        response_5,
    )

    print("\n" + "=" * 70)
    print("END-TO-END RAG TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()