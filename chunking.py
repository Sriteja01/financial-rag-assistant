"""Chunk financial documents while preserving citation metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import tiktoken

from document_builder import FinancialDocument


@dataclass(frozen=True)
class TokenCounter:
    """Count tokens using the encoding associated with an OpenAI model."""

    model_name: str = "gpt-4o-mini"

    def __post_init__(self) -> None:
        try:
            encoding = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        object.__setattr__(self, "encoding", encoding)

    def count(self, text: str) -> int:
        """Return the number of tokens in text."""
        return len(self.encoding.encode(text))


class FinancialChunker:
    """Keep financial rows atomic unless they exceed the token limit."""

    def __init__(
        self,
        chunk_size: int = 250,
        chunk_overlap: int = 30,
        model_name: str = "gpt-4o-mini",
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.token_counter = TokenCounter(model_name=model_name)

    def chunk_text(self, text: str) -> list[str]:
        """Split long text by tokens with overlap between adjacent chunks."""
        tokens = self.token_counter.encoding.encode(text)

        if len(tokens) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.chunk_overlap
        chunks = []

        for start_index in range(0, len(tokens), step):
            token_window = tokens[start_index : start_index + self.chunk_size]

            if not token_window:
                break

            chunks.append(self.token_counter.encoding.decode(token_window))

            if start_index + self.chunk_size >= len(tokens):
                break

        return chunks

    def chunk_documents(
        self,
        documents: Iterable[FinancialDocument],
    ) -> list[FinancialDocument]:
        """Create embedding-ready chunks while retaining full citation metadata."""
        chunked_documents = []

        for document in documents:
            text_chunks = self.chunk_text(document.page_content)
            source_id = (
                f"{document.metadata['source_file']}:"
                f"{document.metadata['source_row']}"
            )

            for chunk_index, chunk_text in enumerate(text_chunks):
                metadata = {
                    **document.metadata,
                    "document_id": source_id,
                    "chunk_index": chunk_index,
                    "chunk_count": len(text_chunks),
                    "token_count": self.token_counter.count(chunk_text),
                }

                chunked_documents.append(
                    FinancialDocument(
                        page_content=chunk_text,
                        metadata=metadata,
                    )
                )

        return chunked_documents