import json
import time
from pathlib import Path

import matplotlib.pyplot as plt

from rag_chatbot.config import Settings
from rag_chatbot.embeddings.embedding_service import EmbeddingService
from rag_chatbot.vectorstore.chroma_store import ChromaVectorStore


EVALUATION_FILE = (
    Settings.DATA_DIR
    / "evaluation"
    / "evaluation_questions.json"
)

OUTPUT_DIR = Settings.PROJECT_ROOT / "outputs"

RESULTS_FILE = OUTPUT_DIR / "retrieval_metrics.json"
CHART_FILE = OUTPUT_DIR / "retrieval_performance.png"


def load_evaluation_questions() -> list[dict]:
    """Load the ground-truth evaluation dataset."""

    with EVALUATION_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def find_expected_source_rank(
    documents,
    expected_source: str,
) -> int | None:
    """
    Return the 1-based rank of the expected source.

    Returns None if the expected source was not retrieved.
    """

    for rank, document in enumerate(
        documents,
        start=1,
    ):
        retrieved_source = document.metadata.get(
            "file_name"
        )

        if retrieved_source == expected_source:
            return rank

    return None


def evaluate_retrieval(
    vector_store: ChromaVectorStore,
    questions: list[dict],
) -> dict:
    """Evaluate retrieval performance across the test dataset."""

    total_questions = len(questions)

    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_5 = 0

    reciprocal_rank_sum = 0.0

    latencies = []

    question_results = []

    for index, item in enumerate(
        questions,
        start=1,
    ):
        question = item["question"]
        expected_source = item["expected_source"]

        print(
            f"\n[{index}/{total_questions}] "
            f"{question}"
        )

        start_time = time.perf_counter()

        documents = vector_store.similarity_search(
            query=question,
            k=5,
        )

        elapsed_time = (
            time.perf_counter()
            - start_time
        )

        latency_ms = elapsed_time * 1000

        latencies.append(latency_ms)

        rank = find_expected_source_rank(
            documents=documents,
            expected_source=expected_source,
        )

        if rank is not None:

            if rank <= 1:
                hit_at_1 += 1

            if rank <= 3:
                hit_at_3 += 1

            if rank <= 5:
                hit_at_5 += 1

            reciprocal_rank_sum += 1 / rank

        retrieved_sources = [
            document.metadata.get(
                "file_name",
                "Unknown",
            )
            for document in documents
        ]

        question_results.append(
            {
                "question": question,
                "expected_source": expected_source,
                "retrieved_sources": retrieved_sources,
                "expected_source_rank": rank,
                "latency_ms": round(
                    latency_ms,
                    2,
                ),
            }
        )

        print(
            f"Expected source: "
            f"{expected_source}"
        )

        print(
            f"Rank: "
            f"{rank if rank is not None else 'Not found'}"
        )

        print(
            f"Latency: "
            f"{latency_ms:.2f} ms"
        )

    metrics = {
        "total_questions": total_questions,

        "hit_rate_at_1": (
            hit_at_1 / total_questions
        ),

        "hit_rate_at_3": (
            hit_at_3 / total_questions
        ),

        "hit_rate_at_5": (
            hit_at_5 / total_questions
        ),

        "mean_reciprocal_rank": (
            reciprocal_rank_sum
            / total_questions
        ),

        "average_retrieval_latency_ms": (
            sum(latencies)
            / len(latencies)
        ),

        "question_results": question_results,
    }

    return metrics


def save_results(
    metrics: dict,
) -> None:
    """Save evaluation results as JSON."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
            ensure_ascii=False,
        )


def create_performance_chart(
    metrics: dict,
) -> None:
    """Generate a retrieval-performance visualization."""

    labels = [
        "Hit Rate@1",
        "Hit Rate@3",
        "Hit Rate@5",
        "MRR",
    ]

    values = [
        metrics["hit_rate_at_1"],
        metrics["hit_rate_at_3"],
        metrics["hit_rate_at_5"],
        metrics["mean_reciprocal_rank"],
    ]

    percentages = [
        value * 100
        for value in values
    ]

    plt.figure(
        figsize=(8, 5)
    )

    bars = plt.bar(
        labels,
        percentages,
    )

    plt.ylim(
        0,
        110,
    )

    plt.ylabel(
        "Score (%)"
    )

    plt.title(
        "RAG Retrieval Performance"
    )

    for bar, percentage in zip(
        bars,
        percentages,
    ):
        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{percentage:.1f}%",
            ha="center",
        )

    plt.tight_layout()

    plt.savefig(
        CHART_FILE,
        dpi=200,
    )

    plt.close()


def print_summary(
    metrics: dict,
) -> None:
    """Display final evaluation metrics."""

    print("\n")
    print("=" * 50)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 50)

    print(
        f"Questions evaluated: "
        f"{metrics['total_questions']}"
    )

    print(
        f"Hit Rate@1: "
        f"{metrics['hit_rate_at_1']:.2%}"
    )

    print(
        f"Hit Rate@3: "
        f"{metrics['hit_rate_at_3']:.2%}"
    )

    print(
        f"Hit Rate@5: "
        f"{metrics['hit_rate_at_5']:.2%}"
    )

    print(
        f"MRR: "
        f"{metrics['mean_reciprocal_rank']:.4f}"
    )

    print(
        f"Average retrieval latency: "
        f"{metrics['average_retrieval_latency_ms']:.2f} ms"
    )

    print("\nSaved:")
    print(RESULTS_FILE)
    print(CHART_FILE)


def main() -> None:
    """Run the complete retrieval evaluation."""

    print("\nLoading evaluation dataset...")

    questions = load_evaluation_questions()

    print(
        f"Loaded {len(questions)} questions."
    )

    print("\nLoading embedding model...")

    embedding_service = EmbeddingService()

    print("Connecting to ChromaDB...")

    vector_store = ChromaVectorStore(
        embedding_function=embedding_service.embeddings
    )

    print("\nRunning evaluation...")

    metrics = evaluate_retrieval(
        vector_store=vector_store,
        questions=questions,
    )

    save_results(metrics)

    create_performance_chart(metrics)

    print_summary(metrics)


if __name__ == "__main__":
    main()