from types import SimpleNamespace

from citations import Citation
from document_builder import FinancialDocument
from evaluator import EvaluationCase, RAGEvaluator
from rag import RAGResponse
from retriever import RetrievalResult


class FakePlainChatModel:
    def invoke(self, input_data):
        return SimpleNamespace(
            content="Retail revenue was $3 million."
        )


class FakeRAGAssistant:
    def answer(self, question):
        document = FinancialDocument(
            page_content=(
                "Retail reported Revenue of $2.15 million during Q1 2024."
            ),
            metadata={
                "source_file": "financials.csv",
                "source_row": 12,
            },
        )

        retrieval_result = RetrievalResult(
            document=document,
            distance=0.08,
            relevance_score=0.93,
            rank=1,
        )

        citation = Citation(
            source_file="financials.csv",
            source_row=12,
            page=None,
            retrieval_score=0.93,
            confidence_label="High",
        )

        return RAGResponse(
            answer=(
                "Retail revenue was $2.15 million.\n\n"
                "Sources:\n"
                "- financials.csv | Row 12"
            ),
            answer_text="Retail revenue was $2.15 million.",
            citations=[citation],
            sources=[retrieval_result],
        )


def test_evaluator_scores_rag_higher_than_plain_llm():
    case = EvaluationCase(
        question="What was Retail revenue?",
        expected_answer_terms=("2.15 million",),
        expected_source_ids=frozenset({"financials.csv:12"}),
        forbidden_answer_terms=("3 million",),
    )

    evaluator = RAGEvaluator(
        rag_assistant=FakeRAGAssistant(),
        plain_chat_model=FakePlainChatModel(),
    )

    results = evaluator.evaluate_cases([case])

    plain_result = results.loc[results["mode"] == "Plain LLM"].iloc[0]
    rag_result = results.loc[results["mode"] == "RAG"].iloc[0]

    assert plain_result["factual_correctness"] == 0.0
    assert plain_result["hallucination_rate"] == 1.0
    assert rag_result["factual_correctness"] == 1.0
    assert rag_result["hallucination_rate"] == 0.0
    assert rag_result["retrieval_precision"] == 1.0
    assert rag_result["retrieval_recall"] == 1.0
    assert rag_result["citation_accuracy"] == 1.0


def test_comparison_table_aggregates_metrics():
    case = EvaluationCase(
        question="What was Retail revenue?",
        expected_answer_terms=("2.15 million",),
        expected_source_ids=frozenset({"financials.csv:12"}),
    )

    evaluator = RAGEvaluator(
        rag_assistant=FakeRAGAssistant(),
        plain_chat_model=FakePlainChatModel(),
    )

    results = evaluator.evaluate_cases([case])
    comparison = evaluator.comparison_table(results)

    assert set(comparison.index) == {"Plain LLM", "RAG"}
    assert comparison.loc["RAG", "Factual Correctness"] == 1.0
    assert comparison.loc["RAG", "Citation Accuracy"] == 1.0

def test_load_evaluation_cases_from_json(tmp_path):
    evaluation_file = tmp_path / "evaluation_cases.json"
    evaluation_file.write_text(
        """[
            {
                "question": "What was revenue?",
                "expected_answer_terms": ["2.15 million"],
                "expected_source_ids": ["financials.csv:2"],
                "forbidden_answer_terms": ["3 million"]
            }
        ]""",
        encoding="utf-8",
    )

    from evaluator import load_evaluation_cases

    cases = load_evaluation_cases(evaluation_file)

    assert len(cases) == 1
    assert cases[0].question == "What was revenue?"
    assert cases[0].expected_source_ids == frozenset(
        {"financials.csv:2"}
    )