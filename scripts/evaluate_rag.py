import json
import re
import statistics
import time
import unicodedata

import matplotlib.pyplot as plt

from rag_chatbot.config import Settings
from rag_chatbot.embeddings.embedding_service import EmbeddingService
from rag_chatbot.llm.llm_service import LLMService
from rag_chatbot.retrieval.retriever import DocumentRetriever
from rag_chatbot.services.chat_service import ChatService
from rag_chatbot.vectorstore.chroma_store import ChromaVectorStore


EVALUATION_FILE = Settings.DATA_DIR / "evaluation" / "evaluation_questions.json"
OUTPUT_DIR = Settings.PROJECT_ROOT / "outputs"
RESULTS_FILE = OUTPUT_DIR / "rag_evaluation_metrics.json"
CHART_FILE = OUTPUT_DIR / "rag_answer_evaluation.png"

ANSWER_MATCH_THRESHOLD = 0.8


def load_questions() -> list[dict]:
    """Load the ground-truth evaluation questions."""
    with EVALUATION_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize_text(text: str) -> str:
    """Normalize answers so minor formatting differences do not count as errors."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("£", "")
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def reference_coverage(generated_answer: str, expected_answer: str) -> float:
    """Measure how much of the reference answer appears in the generated answer."""
    generated = normalize_text(generated_answer)
    expected = normalize_text(expected_answer)

    if expected in generated:
        return 1.0

    expected_tokens = set(expected.split())
    generated_tokens = set(generated.split())

    if not expected_tokens:
        return 0.0

    matches = expected_tokens & generated_tokens
    return len(matches) / len(expected_tokens)


def source_matches(sources: list[str], expected_source: str) -> bool:
    return any(expected_source in source for source in sources)


def build_chat_service() -> ChatService:
    """Set up the RAG components needed for evaluation."""
    print("Loading embedding model...")
    embeddings = EmbeddingService()

    print("Connecting to ChromaDB...")
    vector_store = ChromaVectorStore(
        embedding_function=embeddings.embeddings
    )

    print("Creating retriever...")
    retriever = DocumentRetriever(vector_store)

    print("Connecting to Groq...")
    llm_service = LLMService()

    return ChatService(retriever, llm_service)


def evaluate(chat_service: ChatService, questions: list[dict]) -> dict:
    results = []
    latencies = []
    correct_answers = 0
    correct_sources = 0

    for number, item in enumerate(questions, start=1):
        question = item["question"]
        expected_answer = item["expected_answer"]
        expected_source = item["expected_source"]

        print(f"\n[{number}/{len(questions)}] {question}")

        start = time.perf_counter()
        response = chat_service.answer(question, chat_history=[])
        latency_ms = (time.perf_counter() - start) * 1000

        coverage = reference_coverage(response.answer, expected_answer)
        answer_correct = coverage >= ANSWER_MATCH_THRESHOLD
        source_correct = source_matches(response.sources, expected_source)

        correct_answers += int(answer_correct)
        correct_sources += int(source_correct)
        latencies.append(latency_ms)

        results.append(
            {
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": response.answer,
                "answer_correct": answer_correct,
                "reference_coverage": round(coverage, 4),
                "expected_source": expected_source,
                "returned_sources": response.sources,
                "source_correct": source_correct,
                "retrieval_query": response.retrieval_query,
                "latency_ms": round(latency_ms, 2),
            }
        )

        print(f"Expected: {expected_answer}")
        print(f"Generated: {response.answer}")
        print(f"Answer correct: {answer_correct}")
        print(f"Source correct: {source_correct}")
        print(f"Latency: {latency_ms:.2f} ms")

    total = len(questions)

    return {
        "total_questions": total,
        "answer_accuracy": correct_answers / total,
        "source_attribution_accuracy": correct_sources / total,
        "average_rag_latency_ms": statistics.mean(latencies),
        "median_rag_latency_ms": statistics.median(latencies),
        "question_results": results,
    }


def save_results(metrics: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with RESULTS_FILE.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4, ensure_ascii=False)


def create_chart(metrics: dict) -> None:
    labels = ["Answer Accuracy", "Source Accuracy"]
    values = [
        metrics["answer_accuracy"] * 100,
        metrics["source_attribution_accuracy"] * 100,
    ]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values)
    plt.ylim(0, 110)
    plt.ylabel("Accuracy (%)")
    plt.title("End-to-End RAG Evaluation")

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{value:.1f}%",
            ha="center",
        )

    plt.tight_layout()
    plt.savefig(CHART_FILE, dpi=200)
    plt.close()


def print_summary(metrics: dict) -> None:
    print("\n" + "=" * 55)
    print("END-TO-END RAG EVALUATION RESULTS")
    print("=" * 55)
    print(f"Questions evaluated: {metrics['total_questions']}")
    print(f"Answer accuracy: {metrics['answer_accuracy']:.2%}")
    print(
        "Source attribution accuracy: "
        f"{metrics['source_attribution_accuracy']:.2%}"
    )
    print(f"Average RAG latency: {metrics['average_rag_latency_ms']:.2f} ms")
    print(f"Median RAG latency: {metrics['median_rag_latency_ms']:.2f} ms")
    print(f"\nSaved:\n{RESULTS_FILE}\n{CHART_FILE}")


def main() -> None:
    questions = load_questions()

    print(f"Loaded {len(questions)} evaluation questions.")
    chat_service = build_chat_service()

    print("\nRunning end-to-end RAG evaluation...")
    metrics = evaluate(chat_service, questions)

    save_results(metrics)
    create_chart(metrics)
    print_summary(metrics)


if __name__ == "__main__":
    main()