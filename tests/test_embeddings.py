import pytest

from document_builder import FinancialDocument
from embeddings import (
    EmbeddingGenerationError,
    EmbeddingService,
)


class FakeEmbeddingProvider:
    """Offline deterministic provider for safe unit tests."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            [float(index), float(len(text)), 0.5]
            for index, text in enumerate(texts)
        ]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.5]


class BrokenEmbeddingProvider:
    """Simulates invalid provider responses."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 2.0]]

    def embed_query(self, text: str) -> list[float]:
        return []


@pytest.fixture
def financial_documents():
    return [
        FinancialDocument(
            page_content="Retail reported Revenue of $2.15 million.",
            metadata={
                "source_file": "financials.csv",
                "source_row": 12,
            },
        ),
        FinancialDocument(
            page_content="Retail reported Net Income of $450,000.",
            metadata={
                "source_file": "financials.csv",
                "source_row": 13,
            },
        ),
    ]


def test_embed_documents_preserves_order_and_citations(financial_documents):
    service = EmbeddingService(
        embedding_provider=FakeEmbeddingProvider()
    )

    embedded_documents = service.embed_documents(financial_documents)

    assert len(embedded_documents) == 2
    assert embedded_documents[0].document.metadata["source_row"] == 12
    assert embedded_documents[1].document.metadata["source_row"] == 13
    assert embedded_documents[0].vector[0] == 0.0
    assert embedded_documents[1].vector[0] == 1.0


def test_embed_documents_returns_empty_list_for_empty_input():
    service = EmbeddingService(
        embedding_provider=FakeEmbeddingProvider()
    )

    assert service.embed_documents([]) == []


def test_embed_query_returns_vector():
    service = EmbeddingService(
        embedding_provider=FakeEmbeddingProvider()
    )

    query = "What was Retail revenue?"
    vector = service.embed_query(query)

    assert vector == [float(len(query)), 1.0, 0.5]


def test_embed_query_rejects_empty_text():
    service = EmbeddingService(
        embedding_provider=FakeEmbeddingProvider()
    )

    with pytest.raises(ValueError, match="cannot be empty"):
        service.embed_query("   ")


def test_embed_documents_rejects_wrong_vector_count(financial_documents):
    service = EmbeddingService(
        embedding_provider=BrokenEmbeddingProvider()
    )

    with pytest.raises(
        EmbeddingGenerationError,
        match="Expected 2 vectors but received 1",
    ):
        service.embed_documents(financial_documents)


def test_embed_query_rejects_empty_vector():
    service = EmbeddingService(
        embedding_provider=BrokenEmbeddingProvider()
    )

    with pytest.raises(
        EmbeddingGenerationError,
        match="empty vectors",
    ):
        service.embed_query("What was revenue?")