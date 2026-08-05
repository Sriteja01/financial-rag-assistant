from types import SimpleNamespace

import pytest

from document_builder import FinancialDocument
from rag import FinancialRAGAssistant
from retriever import RetrievalResult


class FakeRetriever:
    """Returns controlled evidence without FAISS or Ollama."""

    def __init__(self, results):
        self.results = results
        self.last_query = None

    def retrieve(self, query, k, max_distance, metadata_filters):
        self.last_query = query
        return self.results


class FakeChatModel:
    """Returns a controlled answer without calling Ollama."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def invoke(self, input_data):
        self.calls.append(input_data)
        return SimpleNamespace(content=self.answer)


@pytest.fixture
def retrieval_results():
    document = FinancialDocument(
        page_content=(
            "Retail reported Revenue of $2.15 million during Q1 2024."
        ),
        metadata={
            "source_file": "financials.csv",
            "source_row": 12,
        },
    )

    return [
        RetrievalResult(
            document=document,
            distance=0.08,
            relevance_score=0.926,
            rank=1,
        )
    ]


def test_build_context_includes_financial_fact_and_citation(
    retrieval_results,
):
    context = FinancialRAGAssistant.build_context(retrieval_results)

    assert "Revenue of $2.15 million" in context
    assert "Source file: financials.csv" in context
    assert "Source row: 12" in context


def test_answer_uses_retrieved_context(retrieval_results):
    fake_retriever = FakeRetriever(retrieval_results)
    fake_chat_model = FakeChatModel(
        "Retail revenue was $2.15 million in Q1 2024."
    )

    assistant = FinancialRAGAssistant(
        retriever=fake_retriever,
        chat_model=fake_chat_model,
    )

    response = assistant.answer("What was Retail revenue?")

    assert response.answer_text == "Retail revenue was $2.15 million in Q1 2024."
    assert "Sources:" in response.answer
    assert "financials.csv | Row 12" in response.answer
    assert response.citations[0].confidence_label == "High"
    assert response.sources == retrieval_results
    assert fake_retriever.last_query == "What was Retail revenue?"
    assert len(fake_chat_model.calls) == 1


def test_answer_returns_safe_response_when_no_context_exists():
    fake_retriever = FakeRetriever([])
    fake_chat_model = FakeChatModel("This should not be used.")

    assistant = FinancialRAGAssistant(
        retriever=fake_retriever,
        chat_model=fake_chat_model,
    )

    response = assistant.answer("What was net income?")

    assert "do not have enough retrieved" in response.answer
    assert response.sources == []
    assert fake_chat_model.calls == []


def test_answer_rejects_empty_question(retrieval_results):
    assistant = FinancialRAGAssistant(
        retriever=FakeRetriever(retrieval_results),
        chat_model=FakeChatModel("Unused"),
    )

    with pytest.raises(ValueError, match="cannot be empty"):
        assistant.answer("   ")