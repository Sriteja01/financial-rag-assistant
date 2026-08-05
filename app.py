"""Streamlit UI for the Financial Statement RAG Assistant."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

import streamlit as st

from chunking import FinancialChunker
from config import settings
from document_builder import FinancialDocumentBuilder
from embeddings import EmbeddingService
from ingest import FinancialDataIngestor
from rag import FinancialRAGAssistant
from retriever import FaissFinancialVectorStore, FinancialRetriever
from utils import logger


RAW_DATA_DIR = settings.data_dir / "raw"
SUPPORTED_EXTENSIONS = {".csv", ".xls", ".xlsx"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB per file


@dataclass(frozen=True)
class IndexBuildStats:
    """Summary of a completed indexing operation."""

    source_files: int
    cleaned_rows: int
    documents: int
    chunks: int
    vector_dimension: int


def init_session_state() -> None:
    """Initialize application state once per Streamlit user session."""
    defaults = {
        "index_built": False,
        "rag_assistant": None,
        "index_stats": None,
        "indexing_error": None,
        "messages": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def save_uploaded_files(uploaded_files) -> list[Path]:
    """Save validated uploads with unique filenames under data/raw/."""
    if not uploaded_files:
        raise ValueError("Upload at least one CSV or Excel file.")

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for uploaded_file in uploaded_files:
        original_name = Path(uploaded_file.name).name
        suffix = Path(original_name).suffix.lower()

        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file '{original_name}'. "
                "Use CSV, XLS, or XLSX files."
            )

        file_size = getattr(uploaded_file, "size", 0)
        if file_size and file_size > MAX_UPLOAD_BYTES:
            raise ValueError(
                f"'{original_name}' is larger than the 50 MB upload limit."
            )

        unique_name = (
            f"{Path(original_name).stem}_{uuid4().hex[:8]}{suffix}"
        )
        target_path = RAW_DATA_DIR / unique_name
        target_path.write_bytes(uploaded_file.getbuffer())
        saved_paths.append(target_path)

        logger.info("Saved uploaded financial file: %s", target_path.name)

    return saved_paths


def build_rag_assistant(
    file_paths: list[Path],
) -> tuple[FinancialRAGAssistant, IndexBuildStats]:
    """Build the actual ingestion-to-RAG pipeline for uploaded files."""
    ingestor = FinancialDataIngestor()
    cleaned_dataframe = ingestor.load_many(file_paths)

    document_builder = FinancialDocumentBuilder()
    documents = document_builder.build_documents(cleaned_dataframe)

    chunker = FinancialChunker(chunk_size=250, chunk_overlap=30)
    chunks = chunker.chunk_documents(documents)

    embedding_service = EmbeddingService()
    embedded_documents = embedding_service.embed_documents(chunks)

    vector_store = FaissFinancialVectorStore(
        embedding_model=embedding_service.model_name
    )
    vector_store.build(embedded_documents)

    retriever = FinancialRetriever(
        vector_store=vector_store,
        embedding_service=embedding_service,
    )
    assistant = FinancialRAGAssistant(retriever=retriever)

    stats = IndexBuildStats(
        source_files=len(file_paths),
        cleaned_rows=len(cleaned_dataframe),
        documents=len(documents),
        chunks=len(chunks),
        vector_dimension=vector_store.dimension or 0,
    )

    logger.info("Built Streamlit RAG index: %s", stats)
    return assistant, stats


def serialize_sources(retrieval_results) -> list[dict]:
    """Convert retrieved records into session-safe display data."""
    sources = []

    for result in retrieval_results:
        metadata = result.document.metadata

        sources.append(
            {
                "rank": result.rank,
                "source_file": metadata.get("source_file", "Unknown"),
                "source_row": metadata.get("source_row", "N/A"),
                "page": metadata.get("page") or metadata.get("source_page"),
                "relevance_score": result.relevance_score,
                "distance": result.distance,
                "page_content": result.document.page_content,
            }
        )

    return sources


def display_retrieved_sources(sources: list[dict]) -> None:
    """Render evidence used for an answer."""
    if not sources:
        return

    with st.expander("View retrieved financial records"):
        st.caption(
            "Retrieval confidence is based on vector similarity. "
            "It is not a factual-certainty score."
        )

        for source in sources:
            citation = (
                f"Source {source['rank']}: {source['source_file']} "
                f"| Row {source['source_row']}"
            )

            if source["page"] is not None:
                citation += f" | Page {source['page']}"

            st.markdown(f"**{citation}**")
            st.caption(
                f"Retrieval score: {source['relevance_score']:.3f} "
                f"| L2 distance: {source['distance']:.4f}"
            )
            st.code(source["page_content"], language="text")


def render_chat_history() -> None:
    """Display the current session's chat history."""
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                display_retrieved_sources(message.get("sources", []))


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(
        page_title="Financial Statement RAG Assistant",
        page_icon="📊",
        layout="wide",
    )

    settings.ensure_directories()
    init_session_state()

    st.title("📊 Financial Statement RAG Assistant")
    st.caption(
        "Ask questions using only the financial records you upload. "
        "All AI processing runs locally through Ollama."
    )

    with st.sidebar:
        st.header("Data and Index")

        uploaded_files = st.file_uploader(
            "Upload financial statements",
            type=["csv", "xls", "xlsx"],
            accept_multiple_files=True,
            help="Upload one or more CSV, XLS, or XLSX financial statements.",
        )

        build_clicked = st.button(
            "Build / Rebuild Index",
            type="primary",
            disabled=not uploaded_files,
        )

        if build_clicked:
            try:
                with st.spinner(
                    "Cleaning files, creating embeddings, and building FAISS index..."
                ):
                    saved_paths = save_uploaded_files(uploaded_files)
                    assistant, stats = build_rag_assistant(saved_paths)

                st.session_state["rag_assistant"] = assistant
                st.session_state["index_stats"] = asdict(stats)
                st.session_state["index_built"] = True
                st.session_state["indexing_error"] = None
                st.session_state["messages"] = []

                st.success("Index built successfully.")

            except Exception as error:
                logger.exception("Streamlit index build failed.")
                st.session_state["index_built"] = False
                st.session_state["rag_assistant"] = None
                st.session_state["indexing_error"] = str(error)

        st.divider()
        st.caption(
            "The in-memory index is available only for this browser session. "
            "Rebuild it after restarting the app."
        )

    if st.session_state["indexing_error"]:
        st.error(
            f"Indexing failed: {st.session_state['indexing_error']}"
        )

    if not st.session_state["index_built"]:
        st.info(
            "Upload financial statements in the sidebar and build an index "
            "to begin asking questions."
        )
        return

    stats = st.session_state["index_stats"]
    columns = st.columns(4)
    columns[0].metric("Index", "Active")
    columns[1].metric("Files", stats["source_files"])
    columns[2].metric("Cleaned rows", stats["cleaned_rows"])
    columns[3].metric("Chunks", stats["chunks"])

    render_chat_history()

    question = st.chat_input(
        "Ask about revenue, net income, cash, assets, or another uploaded metric..."
    )

    if not question:
        return

    st.session_state["messages"].append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching financial records..."):
                response = st.session_state["rag_assistant"].answer(
                    question,
                    k=4,
                )

            sources = serialize_sources(response.sources)
            st.markdown(response.answer)
            display_retrieved_sources(sources)

            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": response.answer,
                    "sources": sources,
                }
            )

        except Exception as error:
            logger.exception("Streamlit RAG question failed.")
            error_message = f"Unable to answer the question: {error}"
            st.error(error_message)

            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "sources": [],
                }
            )


if __name__ == "__main__":
    main()