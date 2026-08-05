"""Generate local Ollama embeddings for financial-document chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from langchain_ollama import OllamaEmbeddings

from config import settings
from document_builder import FinancialDocument
from utils import logger


class EmbeddingProvider(Protocol):
    """Common interface for local or cloud embedding providers."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate vectors for a list of documents."""

    def embed_query(self, text: str) -> list[float]:
        """Generate a vector for one search query."""


class EmbeddingGenerationError(RuntimeError):
    """Raised when the embedding provider fails or returns invalid data."""


@dataclass(frozen=True)
class EmbeddedFinancialDocument:
    """A financial document paired with its generated vector."""

    document: FinancialDocument
    vector: list[float]


class EmbeddingService:
    """Generate local embeddings while preserving source citations."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        model_name: str | None = None,
    ) -> None:
        self.model_name = model_name or settings.embedding_model

        self.embedding_provider = (
            embedding_provider
            if embedding_provider is not None
            else self._create_ollama_provider()
        )

    def _create_ollama_provider(self) -> OllamaEmbeddings:
        """Create LangChain's local Ollama embedding provider."""
        return OllamaEmbeddings(
            model=self.model_name,
            base_url=settings.ollama_base_url,
        )

    def embed_documents(
        self,
        documents: list[FinancialDocument],
    ) -> list[EmbeddedFinancialDocument]:
        """Embed financial chunks in order, retaining all metadata."""
        if not documents:
            return []

        texts = [document.page_content for document in documents]

        try:
            vectors = self.embedding_provider.embed_documents(texts)
        except Exception as error:
            logger.exception(
                "Failed to embed %d financial documents.",
                len(documents),
            )
            raise EmbeddingGenerationError(
                "Could not generate embeddings. Check that Ollama is running "
                f"and model '{self.model_name}' is installed."
            ) from error

        self._validate_vectors(vectors, expected_count=len(documents))

        embedded_documents = [
            EmbeddedFinancialDocument(document=document, vector=vector)
            for document, vector in zip(documents, vectors, strict=True)
        ]

        logger.info(
            "Generated %d local embeddings using %s.",
            len(embedded_documents),
            self.model_name,
        )

        return embedded_documents

    def embed_query(self, query: str) -> list[float]:
        """Embed a user's financial question for similarity search."""
        if not query or not query.strip():
            raise ValueError("Query text cannot be empty.")

        try:
            vector = self.embedding_provider.embed_query(query)
        except Exception as error:
            logger.exception("Failed to embed query.")
            raise EmbeddingGenerationError(
                "Could not embed the query. Check that Ollama is running "
                f"and model '{self.model_name}' is installed."
            ) from error

        self._validate_vectors([vector], expected_count=1)
        return vector

    @staticmethod
    def _validate_vectors(
        vectors: list[list[float]],
        expected_count: int,
    ) -> None:
        """Ensure the provider returned usable, consistent vectors."""
        if len(vectors) != expected_count:
            raise EmbeddingGenerationError(
                f"Expected {expected_count} vectors but received {len(vectors)}."
            )

        if not vectors or not vectors[0]:
            raise EmbeddingGenerationError(
                "Embedding provider returned empty vectors."
            )

        expected_dimension = len(vectors[0])

        if any(len(vector) != expected_dimension for vector in vectors):
            raise EmbeddingGenerationError(
                "Embedding provider returned inconsistent vector dimensions."
            )