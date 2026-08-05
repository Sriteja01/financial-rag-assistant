from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

import app
from document_builder import FinancialDocument
from embeddings import EmbeddedFinancialDocument
from retriever import RetrievalResult


def test_init_session_state():
    session_state = {}

    with patch.object(app, "st") as mock_st:
        mock_st.session_state = session_state

        app.init_session_state()

    assert session_state["index_built"] is False
    assert session_state["rag_assistant"] is None
    assert session_state["index_stats"] is None
    assert session_state["indexing_error"] is None
    assert session_state["messages"] == []


def test_save_uploaded_files_uses_unique_safe_name(tmp_path):
    class DummyUploadedFile:
        name = "../financials.csv"
        size = 10

        @staticmethod
        def getbuffer():
            return b"Line Item,Value\nRevenue,1000\n"

    with patch.object(app, "RAW_DATA_DIR", tmp_path):
        saved_paths = app.save_uploaded_files([DummyUploadedFile()])

    assert len(saved_paths) == 1
    assert saved_paths[0].parent == tmp_path
    assert saved_paths[0].suffix == ".csv"
    assert ".." not in saved_paths[0].name
    assert saved_paths[0].read_bytes() == (
        b"Line Item,Value\nRevenue,1000\n"
    )


@patch("app.FinancialRAGAssistant")
@patch("app.FinancialRetriever")
@patch("app.FaissFinancialVectorStore")
@patch("app.EmbeddingService")
@patch("app.FinancialChunker")
@patch("app.FinancialDocumentBuilder")
@patch("app.FinancialDataIngestor")
def test_build_rag_assistant_uses_real_module_contracts(
    mock_ingestor_class,
    mock_document_builder_class,
    mock_chunker_class,
    mock_embedding_service_class,
    mock_vector_store_class,
    mock_retriever_class,
    mock_rag_assistant_class,
    tmp_path,
):
    dataframe = pd.DataFrame(
        {
            "metric": ["Revenue"],
            "amount": [1000.0],
            "currency": ["USD"],
            "source_file": ["financials.csv"],
            "source_row": [2],
        }
    )

    document = FinancialDocument(
        page_content="The company reported Revenue of $1.00 thousand.",
        metadata={
            "source_file": "financials.csv",
            "source_row": 2,
        },
    )

    embedded_document = EmbeddedFinancialDocument(
        document=document,
        vector=[0.1, 0.2],
    )

    mock_ingestor_class.return_value.load_many.return_value = dataframe
    mock_document_builder_class.return_value.build_documents.return_value = [
        document
    ]
    mock_chunker_class.return_value.chunk_documents.return_value = [document]

    embedding_service = mock_embedding_service_class.return_value
    embedding_service.model_name = "embeddinggemma"
    embedding_service.embed_documents.return_value = [embedded_document]

    vector_store = mock_vector_store_class.return_value
    vector_store.dimension = 2

    file_path = tmp_path / "financials.csv"
    file_path.write_text("placeholder", encoding="utf-8")

    assistant, stats = app.build_rag_assistant([file_path])

    mock_ingestor_class.return_value.load_many.assert_called_once_with(
        [file_path]
    )
    embedding_service.embed_documents.assert_called_once_with([document])
    vector_store.build.assert_called_once_with([embedded_document])
    mock_retriever_class.assert_called_once_with(
        vector_store=vector_store,
        embedding_service=embedding_service,
    )
    mock_rag_assistant_class.assert_called_once()

    assert assistant is mock_rag_assistant_class.return_value
    assert stats.cleaned_rows == 1
    assert stats.documents == 1
    assert stats.chunks == 1
    assert stats.vector_dimension == 2


def test_serialize_sources_preserves_citation_information():
    document = FinancialDocument(
        page_content="Retail reported Revenue of $2.15 million.",
        metadata={
            "source_file": "financials.csv",
            "source_row": 12,
        },
    )

    result = RetrievalResult(
        document=document,
        distance=0.08,
        relevance_score=0.926,
        rank=1,
    )

    sources = app.serialize_sources([result])

    assert sources == [
        {
            "rank": 1,
            "source_file": "financials.csv",
            "source_row": 12,
            "page": None,
            "relevance_score": 0.926,
            "distance": 0.08,
            "page_content": "Retail reported Revenue of $2.15 million.",
        }
    ]