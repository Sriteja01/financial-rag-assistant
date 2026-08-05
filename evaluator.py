
"""Evaluate RAG answers against a plain local LLM baseline."""


import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate

from rag import RAGResponse


@dataclass(frozen=True)
class EvaluationCase:
    """One labeled question used to measure RAG quality."""

    question: str
    expected_answer_terms: tuple[str, ...]
    expected_source_ids: frozenset[str]
    forbidden_answer_terms: tuple[str, ...] = ()


class RAGEvaluator:
    """Compare a grounded RAG pipeline with a plain chat-model baseline."""

    def __init__(
        self,
        rag_assistant: Any,
        plain_chat_model: Any,
    ) -> None:
        self.rag_assistant = rag_assistant
        self.plain_chat_model = plain_chat_model

        self.plain_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Answer the user's financial question in one concise sentence. "
                    "Do not explain your reasoning."
                ),
                ("human", "{question}"),
            ]
        )

    def evaluate_case(self, case: EvaluationCase) -> list[dict[str, Any]]:
        """Evaluate one question using both Plain LLM and RAG."""
        plain_result = self._evaluate_plain_llm(case)
        rag_result = self._evaluate_rag(case)

        return [plain_result, rag_result]

    def evaluate_cases(
        self,
        cases: list[EvaluationCase],
    ) -> pd.DataFrame:
        """Evaluate a labeled dataset and return row-level measurements."""
        rows = []

        for case in cases:
            rows.extend(self.evaluate_case(case))

        return pd.DataFrame(rows)

    @staticmethod
    def comparison_table(results: pd.DataFrame) -> pd.DataFrame:
        """Aggregate row-level measurements into a comparison table."""
        metrics = [
            "factual_correctness",
            "hallucination_rate",
            "latency_ms",
            "retrieval_precision",
            "retrieval_recall",
            "citation_accuracy",
        ]

        return (
            results.groupby("mode")[metrics]
            .mean()
            .round(3)
            .rename(
                columns={
                    "factual_correctness": "Factual Correctness",
                    "hallucination_rate": "Hallucination Rate",
                    "latency_ms": "Average Latency (ms)",
                    "retrieval_precision": "Retrieval Precision",
                    "retrieval_recall": "Retrieval Recall",
                    "citation_accuracy": "Citation Accuracy",
                }
            )
        )

    def _evaluate_plain_llm(
        self,
        case: EvaluationCase,
    ) -> dict[str, Any]:
        """Measure an answer produced without retrieved financial context."""
        messages = self.plain_prompt.format_messages(question=case.question)

        start_time = perf_counter()
        response = self.plain_chat_model.invoke(messages)
        latency_ms = (perf_counter() - start_time) * 1_000

        answer_text = str(response.content).strip()

        return self._score_answer(
            mode="Plain LLM",
            case=case,
            answer_text=answer_text,
            latency_ms=latency_ms,
            retrieved_source_ids=set(),
            cited_source_ids=set(),
        )

    def _evaluate_rag(
        self,
        case: EvaluationCase,
    ) -> dict[str, Any]:
        """Measure the grounded answer and the retrieved source records."""
        start_time = perf_counter()
        response: RAGResponse = self.rag_assistant.answer(case.question)
        latency_ms = (perf_counter() - start_time) * 1_000

        retrieved_source_ids = {
            self._source_id(result.document.metadata)
            for result in response.sources
        }

        cited_source_ids = {
            self._citation_source_id(citation)
            for citation in response.citations
        }

        return self._score_answer(
            mode="RAG",
            case=case,
            answer_text=response.answer_text,
            latency_ms=latency_ms,
            retrieved_source_ids=retrieved_source_ids,
            cited_source_ids=cited_source_ids,
        )

    @staticmethod
    def _score_answer(
        mode: str,
        case: EvaluationCase,
        answer_text: str,
        latency_ms: float,
        retrieved_source_ids: set[str],
        cited_source_ids: set[str],
    ) -> dict[str, Any]:
        """Calculate deterministic quality metrics for one answer."""
        normalized_answer = answer_text.casefold()

        factual_correctness = float(
            all(
                term.casefold() in normalized_answer
                for term in case.expected_answer_terms
            )
        )

        hallucination_rate = float(
            any(
                term.casefold() in normalized_answer
                for term in case.forbidden_answer_terms
            )
        )

        retrieval_precision = RAGEvaluator._set_precision(
            actual=retrieved_source_ids,
            expected=case.expected_source_ids,
        )

        retrieval_recall = RAGEvaluator._set_recall(
            actual=retrieved_source_ids,
            expected=case.expected_source_ids,
        )

        citation_accuracy = RAGEvaluator._set_precision(
            actual=cited_source_ids,
            expected=case.expected_source_ids,
        )

        return {
            "mode": mode,
            "question": case.question,
            "answer": answer_text,
            "factual_correctness": factual_correctness,
            "hallucination_rate": hallucination_rate,
            "latency_ms": latency_ms,
            "retrieval_precision": retrieval_precision,
            "retrieval_recall": retrieval_recall,
            "citation_accuracy": citation_accuracy,
        }

    @staticmethod
    def _source_id(metadata: dict[str, Any]) -> str:
        """Create a stable source identifier from document metadata."""
        return (
            f"{metadata.get('source_file')}:"
            f"{metadata.get('source_row')}"
        )

    @staticmethod
    def _citation_source_id(citation: Any) -> str:
        """Create the same identifier from a formatted citation."""
        return f"{citation.source_file}:{citation.source_row}"

    @staticmethod
    def _set_precision(
        actual: set[str],
        expected: frozenset[str],
    ) -> float:
        """Calculate the fraction of returned records that are correct."""
        if not actual:
            return 0.0

        return len(actual & expected) / len(actual)

    @staticmethod
    def _set_recall(
        actual: set[str],
        expected: frozenset[str],
    ) -> float:
        """Calculate the fraction of expected records that were found."""
        if not expected:
            return 1.0

        return len(actual & expected) / len(expected)

def load_evaluation_cases(
    file_path: str | Path,
) -> list[EvaluationCase]:
    """Load labeled evaluation cases from a JSON file."""
    path = Path(file_path)

    with path.open("r", encoding="utf-8") as file:
        raw_cases = json.load(file)

    if not isinstance(raw_cases, list):
        raise ValueError("Evaluation JSON must contain a list of cases.")

    required_fields = {
        "question",
        "expected_answer_terms",
        "expected_source_ids",
    }

    evaluation_cases = []

    for index, raw_case in enumerate(raw_cases, start=1):
        missing_fields = required_fields - set(raw_case)

        if missing_fields:
            raise ValueError(
                f"Evaluation case {index} is missing fields: "
                f"{sorted(missing_fields)}"
            )

        evaluation_cases.append(
            EvaluationCase(
                question=raw_case["question"],
                expected_answer_terms=tuple(
                    raw_case["expected_answer_terms"]
                ),
                expected_source_ids=frozenset(
                    raw_case["expected_source_ids"]
                ),
                forbidden_answer_terms=tuple(
                    raw_case.get("forbidden_answer_terms", [])
                ),
            )
        )

    return evaluation_cases