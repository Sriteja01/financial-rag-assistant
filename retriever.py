"""Persist and search financial-document embeddings with FAISS."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd

from config import settings
from document_builder import FinancialDocument
from embeddings import EmbeddedFinancialDocument
from utils import logger


INDEX_FILE_NAME = "financial.index"
METADATA_FILE_NAME = "metadata.json"


class VectorStoreError(RuntimeError):
    """Raised when a FAISS vector-store operation cannot be completed."""


@dataclass(frozen=True)
class VectorSearchResult:
    """One nearest-neighbor result from FAISS."""

    document: FinancialDocument
    distance: float
    rank: int


def to_json_safe(value: Any) -> Any:
    """Convert Pandas and NumPy values into JSON-compatible values."""
    if isinstance(value, dict):
        return {key: to_json_safe(item) for key, item in value.items()}

    if isinstance(value, list):
        return [to_json_safe(item) for item in value]

    if isinstance(value, np.generic):
        return value.item()

    if value is None:
        return None

    if not isinstance(value, (str, int, float, bool)) and pd.isna(value):
        return None

    return value


class FaissFinancialVectorStore:
    """A FAISS index plus citation metadata stored without pickle files."""

    def __init__(self, embedding_model: str | None = None) -> None:
        self.embedding_model = embedding_model or settings.embedding_model
        self.index = None
        self.documents: list[FinancialDocument] = []
        self.dimension: int | None = None

    def build(
        self,
        embedded_documents: list[EmbeddedFinancialDocument],
    ) -> None:
        """Create a new exact L2 FAISS index from embedded documents."""
        if not embedded_documents:
            raise VectorStoreError(
                "Cannot build a vector store from zero embedded documents."
            )

        vectors = np.asarray(
            [item.vector for item in embedded_documents],
            dtype=np.float32,
        )

        if vectors.ndim != 2 or vectors.shape[1] == 0:
            raise VectorStoreError("Embeddings must be non-empty 2D vectors.")

        self.dimension = int(vectors.shape[1])
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(vectors)
        self.documents = [
            item.document
            for item in embedded_documents
        ]

        logger.info(
            "Built FAISS IndexFlatL2 with %d vectors and dimension %d.",
            self.index.ntotal,
            self.dimension,
        )

    def search_by_vector(
        self,
        query_vector: list[float],
        k: int = 4,
    ) -> list[VectorSearchResult]:
        """Return the k nearest documents for a query embedding."""
        self._ensure_ready()

        if k <= 0:
            raise ValueError("k must be greater than zero.")

        query = np.asarray([query_vector], dtype=np.float32)

        if query.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension {query.shape[1]} does not match "
                f"index dimension {self.dimension}."
            )

        result_count = min(k, self.index.ntotal)
        distances, indexes = self.index.search(query, result_count)

        results = [
            VectorSearchResult(
                document=self.documents[index],
                distance=float(distance),
                rank=rank + 1,
            )
            for rank, (distance, index) in enumerate(
                zip(distances[0], indexes[0], strict=True)
            )
            if index != -1
        ]

        return results

    def save(self, directory: str | Path) -> None:
        """Save the FAISS index and its aligned document metadata."""
        self._ensure_ready()

        output_directory = Path(directory)
        output_directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(
            self.index,
            str(output_directory / INDEX_FILE_NAME),
        )

        manifest = {
            "embedding_model": self.embedding_model,
            "dimension": self.dimension,
            "documents": [
                {
                    "page_content": document.page_content,
                    "metadata": to_json_safe(document.metadata),
                }
                for document in self.documents
            ],
        }

        with (output_directory / METADATA_FILE_NAME).open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(manifest, file, indent=2, ensure_ascii=False)

        logger.info("Saved FAISS vector store to %s.", output_directory)

    @classmethod
    def load(cls, directory: str | Path) -> "FaissFinancialVectorStore":
        """Load a saved index and verify its metadata alignment."""
        input_directory = Path(directory)
        index_path = input_directory / INDEX_FILE_NAME
        metadata_path = input_directory / METADATA_FILE_NAME

        if not index_path.exists() or not metadata_path.exists():
            raise VectorStoreError(
                "Vector store is incomplete. Both index and metadata files "
                "must exist."
            )

        with metadata_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)

        store = cls(embedding_model=manifest["embedding_model"])
        store.index = faiss.read_index(str(index_path))
        store.dimension = int(manifest["dimension"])
        store.documents = [
            FinancialDocument(
                page_content=item["page_content"],
                metadata=item["metadata"],
            )
            for item in manifest["documents"]
        ]

        if store.index.d != store.dimension:
            raise VectorStoreError(
                "Saved index dimension does not match metadata dimension."
            )

        if store.index.ntotal != len(store.documents):
            raise VectorStoreError(
                "Saved index vector count does not match document metadata."
            )

        logger.info(
            "Loaded FAISS vector store with %d vectors from %s.",
            store.index.ntotal,
            input_directory,
        )

        return store

    def _ensure_ready(self) -> None:
        """Ensure the store was built or loaded before it is used."""
        if self.index is None or self.dimension is None:
            raise VectorStoreError(
                "Vector store is not ready. Build or load an index first."
            )

@dataclass(frozen=True)
class RetrievalResult:
    """A filtered, citation-ready financial retrieval result."""

    document: FinancialDocument
    distance: float
    relevance_score: float
    rank: int


class FinancialRetriever:
    """Embed questions and retrieve relevant financial evidence from FAISS."""

    def __init__(
        self,
        vector_store: FaissFinancialVectorStore,
        embedding_service: Any,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_service = embedding_service

    def retrieve(
        self,
        query: str,
        k: int = 4,
        max_distance: float | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve financial records relevant to a user question.

        Args:
            query: User's natural-language financial question.
            k: Maximum number of records to return.
            max_distance: Optional maximum squared L2 distance.
            metadata_filters: Exact matches, for example:
                {"business_unit": "Retail", "currency": "USD"}.
        """
        if not query or not query.strip():
            raise ValueError("Query text cannot be empty.")

        if k <= 0:
            raise ValueError("k must be greater than zero.")

        if max_distance is not None and max_distance < 0:
            raise ValueError("max_distance cannot be negative.")

        query_vector = self.embedding_service.embed_query(query)

        # When filtering, search the full index first. This prevents relevant
        # filtered records from being missed because unfiltered rows ranked first.
        candidate_count = (
            self.vector_store.index.ntotal
            if metadata_filters
            else k
        )

        candidates = self.vector_store.search_by_vector(
            query_vector=query_vector,
            k=candidate_count,
        )

        retrieved_results = []

        for candidate in candidates:
            if max_distance is not None and candidate.distance > max_distance:
                continue

            if not self._matches_filters(
                candidate.document.metadata,
                metadata_filters,
            ):
                continue

            retrieved_results.append(
                RetrievalResult(
                    document=candidate.document,
                    distance=candidate.distance,
                    relevance_score=self._distance_to_score(
                        candidate.distance
                    ),
                    rank=len(retrieved_results) + 1,
                )
            )

            if len(retrieved_results) == k:
                break

        logger.info(
            "Retrieved %d records for query '%s'.",
            len(retrieved_results),
            query,
        )

        return retrieved_results

    @staticmethod
    def _distance_to_score(distance: float) -> float:
        """Convert L2 distance into a readable score between 0 and 1."""
        return 1 / (1 + distance)

    @staticmethod
    def _matches_filters(
        metadata: dict[str, Any],
        metadata_filters: dict[str, Any] | None,
    ) -> bool:
        """Return True when all supplied metadata filters match."""
        if not metadata_filters:
            return True

        for field, expected_value in metadata_filters.items():
            actual_value = metadata.get(field)

            if actual_value is None:
                return False

            if isinstance(actual_value, str) and isinstance(
                expected_value,
                str,
            ):
                if actual_value.casefold() != expected_value.casefold():
                    return False
            elif actual_value != expected_value:
                return False

        return True