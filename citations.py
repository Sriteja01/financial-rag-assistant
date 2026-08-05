"""Create reliable, programmatic citations from retrieval results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Citation:
    """Traceable evidence used by a RAG answer."""

    source_file: str
    source_row: int | None
    page: int | str | None
    retrieval_score: float
    confidence_label: str


class CitationFormatter:
    """Build and render answer citations from retrieval metadata."""

    def build_citations(
        self,
        retrieval_results: list[Any],
    ) -> list[Citation]:
        """Create one best citation per unique source record."""
        unique_citations: dict[tuple, Citation] = {}

        for result in retrieval_results:
            metadata = result.document.metadata

            citation = Citation(
                source_file=str(metadata.get("source_file", "Unknown source")),
                source_row=metadata.get("source_row"),
                page=metadata.get("page") or metadata.get("source_page"),
                retrieval_score=result.relevance_score,
                confidence_label=self.get_confidence_label(
                    result.relevance_score
                ),
            )

            key = (
                citation.source_file,
                citation.source_row,
                citation.page,
            )

            existing = unique_citations.get(key)

            if (
                existing is None
                or citation.retrieval_score > existing.retrieval_score
            ):
                unique_citations[key] = citation

        return sorted(
            unique_citations.values(),
            key=lambda citation: citation.retrieval_score,
            reverse=True,
        )

    @staticmethod
    def get_confidence_label(retrieval_score: float) -> str:
        """Map a retrieval score to a readable, non-factual confidence label."""
        if retrieval_score >= 0.80:
            return "High"
        if retrieval_score >= 0.50:
            return "Medium"
        return "Low"

    @staticmethod
    def format_answer(
        answer_text: str,
        citations: list[Citation],
    ) -> str:
        """Append source records to an answer in a user-readable format."""
        if not citations:
            return answer_text

        citation_lines = []

        for citation in citations:
            line = f"- {citation.source_file}"

            if citation.source_row is not None:
                line += f" | Row {citation.source_row}"

            if citation.page is not None:
                line += f" | Page {citation.page}"

            line += (
                f" | Retrieval confidence: "
                f"{citation.confidence_label} "
                f"({citation.retrieval_score:.2f})"
            )

            citation_lines.append(line)

        return (
            f"{answer_text}\n\n"
            f"Sources:\n"
            f"{chr(10).join(citation_lines)}"
        )