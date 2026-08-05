import pytest

from chunking import FinancialChunker, TokenCounter
from document_builder import FinancialDocument


def test_token_counter_counts_tokens():
    counter = TokenCounter()

    assert counter.count("Retail reported Revenue of $2.15 million.") > 0


def test_short_document_remains_one_chunk():
    document = FinancialDocument(
        page_content="Retail reported Revenue of $2.15 million during Q1 2024.",
        metadata={
            "source_file": "financials.csv",
            "source_row": 12,
        },
    )

    chunks = FinancialChunker(chunk_size=250).chunk_documents([document])

    assert len(chunks) == 1
    assert chunks[0].page_content == document.page_content
    assert chunks[0].metadata["document_id"] == "financials.csv:12"
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["chunk_count"] == 1


def test_long_document_splits_into_token_limited_chunks():
    document = FinancialDocument(
        page_content="Revenue data " * 100,
        metadata={
            "source_file": "financials.csv",
            "source_row": 25,
        },
    )

    chunker = FinancialChunker(chunk_size=20, chunk_overlap=5)
    chunks = chunker.chunk_documents([document])

    assert len(chunks) > 1
    assert all(chunk.metadata["token_count"] <= 20 for chunk in chunks)
    assert all(
        chunk.metadata["source_file"] == "financials.csv"
        for chunk in chunks
    )
    assert all(
        chunk.metadata["source_row"] == 25
        for chunk in chunks
    )
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(
        range(len(chunks))
    )


def test_adjacent_long_chunks_share_overlap_tokens():
    chunker = FinancialChunker(chunk_size=20, chunk_overlap=5)
    text = "Financial performance analysis " * 100

    chunks = chunker.chunk_text(text)

    first_chunk_tokens = chunker.token_counter.encoding.encode(chunks[0])
    second_chunk_tokens = chunker.token_counter.encoding.encode(chunks[1])

    assert first_chunk_tokens[-5:] == second_chunk_tokens[:5]


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [
        (0, 0),
        (20, -1),
        (20, 20),
    ],
)
def test_invalid_chunk_configuration_is_rejected(
    chunk_size,
    chunk_overlap,
):
    with pytest.raises(ValueError):
        FinancialChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )