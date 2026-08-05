"""Build the sample RAG index and evaluate Plain LLM versus RAG."""

from __future__ import annotations

import argparse

from datetime import datetime

from langchain_ollama import ChatOllama

from chunking import FinancialChunker
from config import settings
from document_builder import FinancialDocumentBuilder
from embeddings import EmbeddingService
from evaluator import RAGEvaluator, load_evaluation_cases
from ingest import FinancialDataIngestor
from rag import FinancialRAGAssistant
from retriever import FaissFinancialVectorStore, FinancialRetriever
from utils import logger


def build_rag_assistant() -> FinancialRAGAssistant:
    """Build the complete local RAG pipeline from the sample CSV."""
    source_file = settings.data_dir / "sample" / "financials.csv"

    ingestor = FinancialDataIngestor()
    cleaned_financials = ingestor.load_file(source_file)

    document_builder = FinancialDocumentBuilder()
    documents = document_builder.build_documents(cleaned_financials)

    chunker = FinancialChunker(chunk_size=250, chunk_overlap=30)
    chunks = chunker.chunk_documents(documents)

    embedding_service = EmbeddingService()
    embedded_documents = embedding_service.embed_documents(chunks)

    vector_store = FaissFinancialVectorStore()
    vector_store.build(embedded_documents)

    evaluation_index_path = (
        settings.vectorstore_dir / "evaluation_sample_index"
    )
    vector_store.save(evaluation_index_path)

    retriever = FinancialRetriever(
        vector_store=vector_store,
        embedding_service=embedding_service,
    )

    return FinancialRAGAssistant(retriever=retriever)


def main() -> None:
    """Run the labeled evaluation and save its results."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N labeled questions.",
    )
    arguments = parser.parse_args()
    settings.ensure_directories()

    evaluation_file = (
        settings.data_dir / "sample" / "evaluation_cases.json"
    )
    evaluation_cases = load_evaluation_cases(evaluation_file)

    if arguments.limit is not None:
        evaluation_cases = evaluation_cases[:arguments.limit]

    rag_assistant = build_rag_assistant()

    plain_chat_model = ChatOllama(
        model=settings.chat_model,
        base_url=settings.ollama_base_url,
        temperature=0,
    )

    evaluator = RAGEvaluator(
        rag_assistant=rag_assistant,
        plain_chat_model=plain_chat_model,
    )

    results = evaluator.evaluate_cases(evaluation_cases)
    comparison = evaluator.comparison_table(results)

    output_directory = settings.data_dir / "processed"
    output_directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_path = output_directory / f"evaluation_results_{timestamp}.csv"
    comparison_path = (
        output_directory / f"evaluation_comparison_{timestamp}.csv"
    )

    results.to_csv(results_path, index=False)
    comparison.to_csv(comparison_path)

    print("\nPlain LLM vs RAG comparison:\n")
    print(comparison.to_string())

    print(f"\nDetailed results saved to: {results_path}")
    print(f"Comparison table saved to: {comparison_path}")

    logger.info(
        "Completed evaluation for %d labeled questions.",
        len(evaluation_cases),
    )


if __name__ == "__main__":
    main()