from document_builder import FinancialDocument
from citations import CitationFormatter
from retriever import RetrievalResult


def test_build_citations_includes_required_provenance():
    document = FinancialDocument(
        page_content="Retail reported Revenue of $2 million.",
        metadata={
            "source_file": "financials.csv",
            "source_row": 12,
            "page": 3,
        },
    )

    result = RetrievalResult(
        document=document,
        distance=0.1,
        relevance_score=0.91,
        rank=1,
    )

    citations = CitationFormatter().build_citations([result])

    assert len(citations) == 1
    assert citations[0].source_file == "financials.csv"
    assert citations[0].source_row == 12
    assert citations[0].page == 3
    assert citations[0].confidence_label == "High"


def test_build_citations_deduplicates_same_source_record():
    document = FinancialDocument(
        page_content="Revenue evidence.",
        metadata={
            "source_file": "financials.csv",
            "source_row": 12,
        },
    )

    weaker_result = RetrievalResult(
        document=document,
        distance=0.4,
        relevance_score=0.60,
        rank=2,
    )

    stronger_result = RetrievalResult(
        document=document,
        distance=0.1,
        relevance_score=0.91,
        rank=1,
    )

    citations = CitationFormatter().build_citations(
        [weaker_result, stronger_result]
    )

    assert len(citations) == 1
    assert citations[0].retrieval_score == 0.91


def test_format_answer_appends_sources():
    formatter = CitationFormatter()

    citation = formatter.build_citations(
        [
            RetrievalResult(
                document=FinancialDocument(
                    page_content="Revenue evidence.",
                    metadata={
                        "source_file": "financials.csv",
                        "source_row": 12,
                    },
                ),
                distance=0.1,
                relevance_score=0.91,
                rank=1,
            )
        ]
    )[0]

    formatted_answer = formatter.format_answer(
        "Retail revenue was $2 million.",
        [citation],
    )

    assert "Sources:" in formatted_answer
    assert "financials.csv | Row 12" in formatted_answer
    assert "High (0.91)" in formatted_answer