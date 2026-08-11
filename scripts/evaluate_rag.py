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


# FILE LOCATIONS

EVALUATION_FILE = (
    Settings.DATA_DIR
    / "evaluation"
    / "evaluation_questions.json"
)

OUTPUT_DIR = Settings.PROJECT_ROOT / "outputs"

RESULTS_FILE = (
    OUTPUT_DIR
    / "rag_evaluation_metrics.json"
)

CHART_FILE = (
    OUTPUT_DIR
    / "rag_answer_evaluation.png"
)


# LOAD EVALUATION DATA

def load_evaluation_questions() -> list[dict]:
    """Load the ground-truth RAG evaluation questions."""

    with EVALUATION_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ANSWER NORMALIZATION

def normalize_text(text: str) -> str:
    """
    Normalize text before comparing generated
    answers against reference answers.
    """

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = text.lower()

    # £1,299 -> 1299
    text = text.replace("£", "")

    text = re.sub(
        r"(?<=\d),(?=\d)",
        "",
        text,
    )

    # Remove punctuation while preserving words/numbers
    text = re.sub(
        r"[^a-z0-9\s-]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def calculate_reference_coverage(
    generated_answer: str,
    expected_answer: str,
) -> float:
    """
    Calculate how much of the expected answer
    appears in the generated response.

    Returns a value between 0 and 1.
    """

    generated = normalize_text(
        generated_answer
    )

    expected = normalize_text(
        expected_answer
    )

    # Exact normalized phrase match
    if expected in generated:
        return 1.0

    expected_tokens = set(
        expected.split()
    )

    generated_tokens = set(
        generated.split()
    )

    if not expected_tokens:
        return 0.0

    matched_tokens = (
        expected_tokens
        & generated_tokens
    )

    return (
        len(matched_tokens)
        / len(expected_tokens)
    )


def is_answer_correct(
    generated_answer: str,
    expected_answer: str,
) -> tuple[bool, float]:
    """
    Determine whether an answer matches
    the ground-truth reference.

    A score of 0.8 or greater is treated
    as correct.
    """

    coverage = calculate_reference_coverage(
        generated_answer,
        expected_answer,
    )

    return coverage >= 0.8, coverage


# SOURCE VALIDATION

def contains_expected_source(
    sources: list[str],
    expected_source: str,
) -> bool:
    """Check whether the expected source was cited."""

    return any(
        expected_source in source
        for source in sources
    )


# RAG INITIALIZATION

def build_chat_service() -> ChatService:
    """Initialize the complete RAG backend."""

    print(
        "Loading embedding model..."
    )

    embedding_service = (
        EmbeddingService()
    )

    print(
        "Connecting to ChromaDB..."
    )

    vector_store = ChromaVectorStore(
        embedding_function=(
            embedding_service.embeddings
        )
    )

    print(
        "Creating retriever..."
    )

    retriever = DocumentRetriever(
        vector_store=vector_store
    )

    print(
        "Connecting to Groq..."
    )

    llm_service = LLMService()

    return ChatService(
        retriever=retriever,
        llm_service=llm_service,
    )


# EVALUATION


def evaluate_rag(
    chat_service: ChatService,
    questions: list[dict],
) -> dict:
    """Evaluate final RAG answer generation."""

    total_questions = len(
        questions
    )

    correct_answers = 0
    correct_sources = 0

    latencies = []

    question_results = []

    for index, item in enumerate(
        questions,
        start=1,
    ):

        question = item[
            "question"
        ]

        expected_answer = item[
            "expected_answer"
        ]

        expected_source = item[
            "expected_source"
        ]

        print("\n" + "=" * 70)

        print(
            f"[{index}/{total_questions}] "
            f"{question}"
        )

        # Run complete RAG request

        start_time = (
            time.perf_counter()
        )

        response = chat_service.answer(
            question=question,
            chat_history=[],
        )

        elapsed_time = (
            time.perf_counter()
            - start_time
        )

        latency_ms = (
            elapsed_time
            * 1000
        )

        latencies.append(
            latency_ms
        )

        # Evaluate answer

        (
            answer_correct,
            reference_coverage,
        ) = is_answer_correct(
            generated_answer=(
                response.answer
            ),
            expected_answer=(
                expected_answer
            ),
        )

        if answer_correct:
            correct_answers += 1

        # Evaluate source attribution

        source_correct = (
            contains_expected_source(
                sources=response.sources,
                expected_source=(
                    expected_source
                ),
            )
        )

        if source_correct:
            correct_sources += 1

        # Save individual result

        result = {
            "question": question,
            "expected_answer": (
                expected_answer
            ),
            "generated_answer": (
                response.answer
            ),
            "answer_correct": (
                answer_correct
            ),
            "reference_coverage": round(
                reference_coverage,
                4,
            ),
            "expected_source": (
                expected_source
            ),
            "returned_sources": (
                response.sources
            ),
            "source_correct": (
                source_correct
            ),
            "retrieval_query": (
                response.retrieval_query
            ),
            "latency_ms": round(
                latency_ms,
                2,
            ),
        }

        question_results.append(
            result
        )

        # Console output

        print(
            f"\nExpected answer: "
            f"{expected_answer}"
        )

        print(
            f"Generated answer: "
            f"{response.answer}"
        )

        print(
            f"Answer correct: "
            f"{answer_correct}"
        )

        print(
            f"Reference coverage: "
            f"{reference_coverage:.2%}"
        )

        print(
            f"Expected source: "
            f"{expected_source}"
        )

        print(
            f"Source correct: "
            f"{source_correct}"
        )

        print(
            f"Latency: "
            f"{latency_ms:.2f} ms"
        )

    # AGGREGATE METRICS

    metrics = {
        "total_questions": (
            total_questions
        ),

        "answer_accuracy": (
            correct_answers
            / total_questions
        ),

        "source_attribution_accuracy": (
            correct_sources
            / total_questions
        ),

        "average_rag_latency_ms": (
            sum(latencies)
            / len(latencies)
        ),

        "median_rag_latency_ms": (
            statistics.median(
                latencies
            )
        ),

        "question_results": (
            question_results
        ),
    }

    return metrics


# SAVE RESULTS

def save_results(
    metrics: dict,
) -> None:
    """Save detailed RAG metrics to JSON."""

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


# VISUALIZATION


def create_chart(
    metrics: dict,
) -> None:
    """Create final answer-quality visualization."""

    labels = [
        "Answer Accuracy",
        "Source Accuracy",
    ]

    values = [
        (
            metrics[
                "answer_accuracy"
            ]
            * 100
        ),
        (
            metrics[
                "source_attribution_accuracy"
            ]
            * 100
        ),
    ]

    plt.figure(
        figsize=(8, 5)
    )

    bars = plt.bar(
        labels,
        values,
    )

    plt.ylim(
        0,
        110,
    )

    plt.ylabel(
        "Accuracy (%)"
    )

    plt.title(
        "End-to-End RAG Evaluation"
    )

    for bar, value in zip(
        bars,
        values,
    ):
        plt.text(
            (
                bar.get_x()
                + bar.get_width()
                / 2
            ),
            bar.get_height() + 2,
            f"{value:.1f}%",
            ha="center",
        )

    plt.tight_layout()

    plt.savefig(
        CHART_FILE,
        dpi=200,
    )

    plt.close()


# SUMMARY

def print_summary(
    metrics: dict,
) -> None:
    """Print final RAG evaluation metrics."""

    print("\n")
    print("=" * 60)
    print("END-TO-END RAG EVALUATION RESULTS")
    print("=" * 60)

    print(
        f"Questions evaluated: "
        f"{metrics['total_questions']}"
    )

    print(
        f"Answer accuracy: "
        f"{metrics['answer_accuracy']:.2%}"
    )

    print(
        f"Source attribution accuracy: "
        f"{metrics['source_attribution_accuracy']:.2%}"
    )

    print(
        f"Average RAG latency: "
        f"{metrics['average_rag_latency_ms']:.2f} ms"
    )

    print(
        f"Median RAG latency: "
        f"{metrics['median_rag_latency_ms']:.2f} ms"
    )

    print("\nSaved:")
    print(RESULTS_FILE)
    print(CHART_FILE)


# MAIN

def main() -> None:
    """Run complete end-to-end RAG evaluation."""

    print("\n")
    print("=" * 60)
    print("INITIALIZING RAG EVALUATION")
    print("=" * 60)

    questions = (
        load_evaluation_questions()
    )

    print(
        f"\nLoaded "
        f"{len(questions)} "
        f"evaluation questions."
    )

    chat_service = (
        build_chat_service()
    )

    print(
        "\nRunning end-to-end "
        "RAG evaluation..."
    )

    metrics = evaluate_rag(
        chat_service=chat_service,
        questions=questions,
    )

    save_results(
        metrics
    )

    create_chart(
        metrics
    )

    print_summary(
        metrics
    )


if __name__ == "__main__":
    main()