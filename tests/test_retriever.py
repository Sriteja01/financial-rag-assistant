import pytest

from document_builder import FinancialDocument
from embeddings import EmbeddedFinancialDocument
from retriever import (
    FaissFinancialVectorStore,
    FinancialRetriever,
    VectorStoreError,
)


@pytest.fixture
def embedded_documents():
    revenue_document = FinancialDocument(
        page_content="Retail reported Revenue of $2.15 million.",
        metadata={
            "source_file": "income_statement.csv",
            "source_row": 12,
        },
    )

    net_income_document = FinancialDocument(
        page_content="Retail reported Net Income of $450,000.",
        metadata={
            "source_file": "income_statement.csv",
            "source_row": 13,
        },
    )

    return [
        EmbeddedFinancialDocument(
            document=revenue_document,
            vector=[1.0, 0.0, 0.0],
        ),
        EmbeddedFinancialDocument(
            document=net_income_document,
            vector=[0.0, 1.0, 0.0],
        ),
    ]


def test_build_creates_exact_faiss_index(embedded_documents):
    store = FaissFinancialVectorStore()
    store.build(embedded_documents)

    assert store.index.ntotal == 2
    assert store.dimension == 3


def test_search_returns_nearest_financial_document(embedded_documents):
    store = FaissFinancialVectorStore()
    store.build(embedded_documents)

    results = store.search_by_vector([0.9, 0.1, 0.0], k=1)

    assert len(results) == 1
    assert results[0].rank == 1
    assert results[0].document.metadata["source_row"] == 12
    assert "Revenue" in results[0].document.page_content


def test_save_and_load_preserves_vectors_and_citations(
    embedded_documents,
    tmp_path,
):
    store = FaissFinancialVectorStore(embedding_model="embeddinggemma")
    store.build(embedded_documents)
    store.save(tmp_path)

    loaded_store = FaissFinancialVectorStore.load(tmp_path)
    results = loaded_store.search_by_vector([0.0, 0.9, 0.0], k=1)

    assert loaded_store.embedding_model == "embeddinggemma"
    assert loaded_store.dimension == 3
    assert results[0].document.metadata["source_file"] == (
        "income_statement.csv"
    )
    assert results[0].document.metadata["source_row"] == 13


def test_build_rejects_empty_documents():
    store = FaissFinancialVectorStore()

    with pytest.raises(VectorStoreError, match="zero embedded documents"):
        store.build([])


def test_search_rejects_wrong_query_dimension(embedded_documents):
    store = FaissFinancialVectorStore()
    store.build(embedded_documents)

    with pytest.raises(ValueError, match="does not match"):
        store.search_by_vector([1.0, 0.0], k=1)


def test_load_rejects_missing_index_files(tmp_path):
    with pytest.raises(VectorStoreError, match="incomplete"):
        FaissFinancialVectorStore.load(tmp_path)

class FixedQueryEmbeddingService:
    """A predictable query embedder used for retrieval tests."""

    def __init__(self, vector):
        self.vector = vector

    def embed_query(self, query):
        return self.vector


def test_retriever_returns_ranked_results(embedded_documents):
    store = FaissFinancialVectorStore()
    store.build(embedded_documents)

    retriever = FinancialRetriever(
        vector_store=store,
        embedding_service=FixedQueryEmbeddingService([0.9, 0.1, 0.0]),
    )

    results = retriever.retrieve("What was revenue?", k=2)

    assert len(results) == 2
    assert results[0].rank == 1
    assert "Revenue" in results[0].document.page_content
    assert 0 < results[0].relevance_score <= 1


def test_retriever_applies_distance_threshold(embedded_documents):
    store = FaissFinancialVectorStore()
    store.build(embedded_documents)

    retriever = FinancialRetriever(
        vector_store=store,
        embedding_service=FixedQueryEmbeddingService([0.9, 0.1, 0.0]),
    )

    results = retriever.retrieve(
        "What was revenue?",
        max_distance=0.001,
    )

    assert results == []


def test_retriever_applies_metadata_filters():
    retail_document = FinancialDocument(
        page_content="Retail reported Revenue of $2 million.",
        metadata={
            "business_unit": "Retail",
            "source_file": "financials.csv",
            "source_row": 2,
        },
    )

    corporate_document = FinancialDocument(
        page_content="Corporate reported Revenue of $1 million.",
        metadata={
            "business_unit": "Corporate",
            "source_file": "financials.csv",
            "source_row": 3,
        },
    )

    store = FaissFinancialVectorStore()
    store.build(
        [
            EmbeddedFinancialDocument(
                document=retail_document,
                vector=[1.0, 0.0],
            ),
            EmbeddedFinancialDocument(
                document=corporate_document,
                vector=[0.0, 1.0],
            ),
        ]
    )

    retriever = FinancialRetriever(
        vector_store=store,
        embedding_service=FixedQueryEmbeddingService([0.9, 0.1]),
    )

    results = retriever.retrieve(
        "What was revenue?",
        metadata_filters={"business_unit": "corporate"},
    )

    assert len(results) == 1
    assert results[0].document.metadata["business_unit"] == "Corporate"


@pytest.mark.parametrize(
    ("query", "k", "max_distance"),
    [
        ("", 4, None),
        ("Revenue", 0, None),
        ("Revenue", 4, -0.1),
    ],
)
def test_retriever_rejects_invalid_arguments(
    embedded_documents,
    query,
    k,
    max_distance,
):
    store = FaissFinancialVectorStore()
    store.build(embedded_documents)

    retriever = FinancialRetriever(
        vector_store=store,
        embedding_service=FixedQueryEmbeddingService([1.0, 0.0, 0.0]),
    )

    with pytest.raises(ValueError):
        retriever.retrieve(
            query=query,
            k=k,
            max_distance=max_distance,
        )