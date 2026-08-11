from collections import Counter

from rag_chatbot.ingestion.ingestion_service import DocumentIngestionService
from rag_chatbot.processing.text_splitter import DocumentTextSplitter


def main() -> None:
    """Test the complete document ingestion and chunking pipeline."""

    # 1. Load all supported documents

    ingestion_service = DocumentIngestionService()

    documents = ingestion_service.load_all()

    print("\n==============================")
    print("DOCUMENT INGESTION RESULTS")
    print("==============================")

    print(f"\nTotal documents loaded: {len(documents)}")


    
    # 2. Count Documents produced from each source file

    document_counts = Counter(
        document.metadata["file_name"]
        for document in documents
    )

    print("\nDocuments created per file:")

    for file_name, count in document_counts.items():
        print(f"  {file_name}: {count}")


    
    # 3. Inspect loaded Documents
    

    print("\n==============================")
    print("DOCUMENT PREVIEW")
    print("==============================")

    for index, document in enumerate(documents, start=1):

        print(f"\nDocument {index}")

        print("Metadata:")
        print(document.metadata)

        print("Content preview:")

        preview = document.page_content[:200]

        print(preview)

        if len(document.page_content) > 200:
            print("...")


    
    # 4. Split Documents into chunks
    

    splitter = DocumentTextSplitter()

    chunks = splitter.split(documents)

    print("\n==============================")
    print("CHUNKING RESULTS")
    print("==============================")

    print(f"\nOriginal Documents: {len(documents)}")
    print(f"Total Chunks: {len(chunks)}")


    # 5. Count chunks by source

    chunk_counts = Counter(
        chunk.metadata["file_name"]
        for chunk in chunks
    )

    print("\nChunks created per file:")

    for file_name, count in chunk_counts.items():
        print(f"  {file_name}: {count}")


    
    # 6. Inspect chunks

    print("\n==============================")
    print("CHUNK PREVIEW")
    print("==============================")

    for index, chunk in enumerate(chunks, start=1):

        print(f"\n--- CHUNK {index} ---")

        print("Metadata:")
        print(chunk.metadata)

        print("Content:")
        print(chunk.page_content)


if __name__ == "__main__":
    main()