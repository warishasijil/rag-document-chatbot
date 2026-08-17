import json
import time

import matplotlib.pyplot as plt

from rag_chatbot.config import Settings
from rag_chatbot.embeddings.embedding_service import EmbeddingService
from rag_chatbot.vectorstore.chroma_store import ChromaVectorStore


EVALUATION_FILE = Settings.DATA_DIR / "evaluation" / "evaluation_questions.json"
OUTPUT_DIR = Settings.PROJECT_ROOT / "outputs"

METRICS_FILE = OUTPUT_DIR / "retrieval_metrics.json"
CHART_FILE = OUTPUT_DIR / "retrieval_performance.png"

TOP_K = 5


def load_questions() -> list[dict]:
    with EVALUATION_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def find_expected_rank(documents, expected_source: str) -> int | None:
    for rank, document in enumerate(documents, start=1):
        if document.metadata.get("file_name") == expected_source:
            return rank

    return None


def evaluate(vector_store: ChromaVectorStore, questions: list[dict]) -> dict:
    results = []
    latencies = []

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    reciprocal_rank_total = 0.0

    for number, item in enumerate(questions, start=1):
        question = item["question"]
        expected_source = item["expected_source"]

        start = time.perf_counter()
        documents = vector_store.similarity_search(question, k=TOP_K)
        latency_ms = (time.perf_counter() - start) * 1000

        rank = find_expected_rank(documents, expected_source)
        latencies.append(latency_ms)

        if rank is not None:
            hits_at_1 += int(rank <= 1)
            hits_at_3 += int(rank <= 3)
            hits_at_5 += int(rank <= 5)
            reciprocal_rank_total += 1 / rank

        retrieved_sources = [
            document.metadata.get("file_name", "Unknown source")
            for document in documents
        ]

        results.append(
            {
                "question": question,
                "expected_source": expected_source,
                "rank": rank,
                "retrieved_sources": retrieved_sources,
                "latency_ms": round(latency_ms, 2),
            }
        )

        print(
            f"[{number}/{len(questions)}] "
            f"rank={rank} | {latency_ms:.2f} ms | {question}"
        )

    total = len(questions)

    return {
        "total_questions": total,
        "hit_rate_at_1": hits_at_1 / total,
        "hit_rate_at_3": hits_at_3 / total,
        "hit_rate_at_5": hits_at_5 / total,
        "mean_reciprocal_rank": reciprocal_rank_total / total,
        "average_retrieval_latency_ms": sum(latencies) / len(latencies),
        "question_results": results,
    }


def save_metrics(metrics: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with METRICS_FILE.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4, ensure_ascii=False)


def create_chart(metrics: dict) -> None:
    labels = ["Hit@1", "Hit@3", "Hit@5", "MRR"]
    values = [
        metrics["hit_rate_at_1"] * 100,
        metrics["hit_rate_at_3"] * 100,
        metrics["hit_rate_at_5"] * 100,
        metrics["mean_reciprocal_rank"] * 100,
    ]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values)

    plt.ylim(0, 110)
    plt.ylabel("Score (%)")
    plt.title("Retrieval Performance")

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
    print("\n" + "=" * 50)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 50)
    print(f"Questions evaluated: {metrics['total_questions']}")
    print(f"Hit Rate@1: {metrics['hit_rate_at_1']:.2%}")
    print(f"Hit Rate@3: {metrics['hit_rate_at_3']:.2%}")
    print(f"Hit Rate@5: {metrics['hit_rate_at_5']:.2%}")
    print(f"MRR: {metrics['mean_reciprocal_rank']:.4f}")
    print(
        "Average retrieval latency: "
        f"{metrics['average_retrieval_latency_ms']:.2f} ms"
    )
    print(f"\nSaved:\n{METRICS_FILE}\n{CHART_FILE}")


def main() -> None:
    questions = load_questions()
    print(f"Loaded {len(questions)} evaluation questions.")

    print("Loading embedding model...")
    embeddings = EmbeddingService()

    print("Connecting to ChromaDB...")
    vector_store = ChromaVectorStore(
        embedding_function=embeddings.embeddings
    )

    print("\nRunning retrieval evaluation...\n")
    metrics = evaluate(vector_store, questions)

    save_metrics(metrics)
    create_chart(metrics)
    print_summary(metrics)


if __name__ == "__main__":
    main()