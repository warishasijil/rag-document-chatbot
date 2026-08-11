"""Validate that the main RAG application modules are implemented."""

import sys


def check_import(description, import_function):
    """Run one import test and report the result."""

    try:
        import_function()
        print(f"[OK]   {description}")
        return True

    except Exception as error:
        print(f"[FAIL] {description}")
        print(f"       {type(error).__name__}: {error}")
        return False


def main():
    print("\n======================================")
    print("RAG PROJECT BACKEND VALIDATION")
    print("======================================\n")

    checks = []

    # --------------------------------------------------
    # Configuration
    # --------------------------------------------------

    checks.append(
        check_import(
            "Settings",
            lambda: __import__(
                "rag_chatbot.config",
                fromlist=["Settings"],
            ).Settings,
        )
    )

    # --------------------------------------------------
    # Document ingestion
    # --------------------------------------------------

    checks.append(
        check_import(
            "BaseDocumentLoader",
            lambda: __import__(
                "rag_chatbot.ingestion.base_loader",
                fromlist=["BaseDocumentLoader"],
            ).BaseDocumentLoader,
        )
    )

    checks.append(
        check_import(
            "TextDocumentLoader",
            lambda: __import__(
                "rag_chatbot.ingestion.text_loader",
                fromlist=["TextDocumentLoader"],
            ).TextDocumentLoader,
        )
    )

    checks.append(
        check_import(
            "PDFDocumentLoader",
            lambda: __import__(
                "rag_chatbot.ingestion.pdf_loader",
                fromlist=["PDFDocumentLoader"],
            ).PDFDocumentLoader,
        )
    )

    checks.append(
        check_import(
            "DOCXDocumentLoader",
            lambda: __import__(
                "rag_chatbot.ingestion.docx_loader",
                fromlist=["DOCXDocumentLoader"],
            ).DOCXDocumentLoader,
        )
    )

    checks.append(
        check_import(
            "ExcelDocumentLoader",
            lambda: __import__(
                "rag_chatbot.ingestion.excel_loader",
                fromlist=["ExcelDocumentLoader"],
            ).ExcelDocumentLoader,
        )
    )

    checks.append(
        check_import(
            "DocumentLoaderFactory",
            lambda: __import__(
                "rag_chatbot.ingestion.loader_factory",
                fromlist=["DocumentLoaderFactory"],
            ).DocumentLoaderFactory,
        )
    )

    checks.append(
        check_import(
            "DocumentIngestionService",
            lambda: __import__(
                "rag_chatbot.ingestion.ingestion_service",
                fromlist=["DocumentIngestionService"],
            ).DocumentIngestionService,
        )
    )

    # --------------------------------------------------
    # Processing
    # --------------------------------------------------

    checks.append(
        check_import(
            "DocumentTextSplitter",
            lambda: __import__(
                "rag_chatbot.processing.text_splitter",
                fromlist=["DocumentTextSplitter"],
            ).DocumentTextSplitter,
        )
    )

    # --------------------------------------------------
    # Embeddings
    # --------------------------------------------------

    checks.append(
        check_import(
            "EmbeddingService",
            lambda: __import__(
                "rag_chatbot.embeddings.embedding_service",
                fromlist=["EmbeddingService"],
            ).EmbeddingService,
        )
    )

    # --------------------------------------------------
    # Vector database
    # --------------------------------------------------

    checks.append(
        check_import(
            "ChromaVectorStore",
            lambda: __import__(
                "rag_chatbot.vectorstore.chroma_store",
                fromlist=["ChromaVectorStore"],
            ).ChromaVectorStore,
        )
    )

    # --------------------------------------------------
    # Retrieval
    # --------------------------------------------------

    checks.append(
        check_import(
            "DocumentRetriever",
            lambda: __import__(
                "rag_chatbot.retrieval.retriever",
                fromlist=["DocumentRetriever"],
            ).DocumentRetriever,
        )
    )

    # --------------------------------------------------
    # LLM
    # --------------------------------------------------

    checks.append(
        check_import(
            "LLMService",
            lambda: __import__(
                "rag_chatbot.llm.llm_service",
                fromlist=["LLMService"],
            ).LLMService,
        )
    )

    # --------------------------------------------------
    # Prompts
    # --------------------------------------------------

    checks.append(
        check_import(
            "RAG prompts",
            lambda: (
                getattr(
                    __import__(
                        "rag_chatbot.prompts.rag_prompt",
                        fromlist=["RAG_PROMPT"],
                    ),
                    "RAG_PROMPT",
                ),
                getattr(
                    __import__(
                        "rag_chatbot.prompts.rag_prompt",
                        fromlist=["CONTEXTUALIZE_PROMPT"],
                    ),
                    "CONTEXTUALIZE_PROMPT",
                ),
            ),
        )
    )

    # --------------------------------------------------
    # Chat service
    # --------------------------------------------------

    checks.append(
        check_import(
            "ChatService",
            lambda: __import__(
                "rag_chatbot.services.chat_service",
                fromlist=["ChatService"],
            ).ChatService,
        )
    )

    print("\n======================================")

    passed = sum(checks)
    total = len(checks)

    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\nAll backend imports are working.")
        print("Safe to proceed to runtime tests.")
        sys.exit(0)

    print("\nSome modules still need to be fixed.")
    sys.exit(1)


if __name__ == "__main__":
    main()