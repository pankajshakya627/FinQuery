"""
Tests for the RAG pipeline components.
Run: pytest tests/ -v
"""

import pytest
from app.models.schemas import (
    QueryRequest, DocumentMetadata, IndexingConfig,
    ChunkStrategy, DocumentStatus, RetrievalStats,
)
from app.rag.chunker import DocumentChunker
from app.rag.parser import DocumentParser
from langchain.schema import Document as LCDocument


# ═══════════════════════════════════════════════════════════════
#  PYDANTIC MODEL TESTS
# ═══════════════════════════════════════════════════════════════

class TestPydanticModels:
    def test_query_request_valid(self):
        req = QueryRequest(question="What is the annual fee for Infinia?")
        assert req.question == "What is the annual fee for Infinia?"
        assert req.top_k == 5
        assert req.similarity_threshold == 0.3
        assert req.include_sources is True

    def test_query_request_min_length(self):
        with pytest.raises(Exception):
            QueryRequest(question="ab")

    def test_query_request_custom_params(self):
        req = QueryRequest(
            question="How does fuel surcharge waiver work?",
            top_k=10,
            similarity_threshold=0.5,
            include_sources=False,
        )
        assert req.top_k == 10
        assert req.similarity_threshold == 0.5
        assert req.include_sources is False

    def test_document_metadata(self):
        meta = DocumentMetadata(
            title="HDFC MITC",
            filename="mitc.pdf",
            file_type="pdf",
            file_size_bytes=543520,
            tags=["credit card", "MITC"],
        )
        assert meta.title == "HDFC MITC"
        assert meta.status == DocumentStatus.PENDING
        assert len(meta.id) == 36  # UUID
        assert meta.tags == ["credit card", "MITC"]

    def test_indexing_config_defaults(self):
        config = IndexingConfig()
        assert config.chunk_strategy == ChunkStrategy.RECURSIVE
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50

    def test_retrieval_stats(self):
        stats = RetrievalStats(
            total_chunks_searched=40,
            chunks_retrieved=5,
            top_k_used=5,
            avg_relevance_score=0.72,
            max_relevance_score=0.91,
            min_relevance_score=0.45,
        )
        assert stats.total_chunks_searched == 40
        assert stats.avg_relevance_score == 0.72


# ═══════════════════════════════════════════════════════════════
#  CHUNKER TESTS
# ═══════════════════════════════════════════════════════════════

class TestDocumentChunker:
    def test_basic_chunking(self):
        chunker = DocumentChunker(
            strategy=ChunkStrategy.RECURSIVE,
            chunk_size=200,
            chunk_overlap=20,
        )
        docs = [
            LCDocument(
                page_content="This is a test document about HDFC credit card fees. " * 20,
                metadata={"page_number": 1, "source_file": "test.pdf"},
            )
        ]
        chunks = chunker.chunk_documents(docs, "test-doc-id")
        assert len(chunks) > 1
        assert all(c.metadata["document_id"] == "test-doc-id" for c in chunks)

    def test_chunk_metadata_enrichment(self):
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        docs = [
            LCDocument(
                page_content="1. Fees and Charges\nThe annual fee for Infinia is Rs 10,000.",
                metadata={"page_number": 1, "source_file": "mitc.pdf"},
            )
        ]
        chunks = chunker.chunk_documents(docs, "doc-123")
        chunk = chunks[0]
        assert "chunk_id" in chunk.metadata
        assert chunk.metadata["document_id"] == "doc-123"
        assert chunk.metadata["chunk_index"] == 0
        assert chunk.metadata["token_count"] > 0

    def test_keyword_extraction(self):
        keywords = DocumentChunker._extract_keywords(
            "The annual fee for HDFC Infinia credit card is Rs 10000. "
            "Fee waiver available on spending Rs 8 lakh annually."
        )
        assert len(keywords) > 0
        assert "annual" in keywords or "fee" in keywords

    def test_section_title_extraction(self):
        title = DocumentChunker._extract_section_title(
            "1. Fees and Charges\nThe annual fee is..."
        )
        assert title is not None
        assert "Fees" in title

    def test_table_detection(self):
        assert DocumentChunker._has_table(
            "Card Type | Fee\n---|---\nInfinia | Rs 10,000\nRegalia | Rs 2,500\nMillennia | Rs 1,000"
        )
        assert not DocumentChunker._has_table("This is a normal paragraph without tables.")

    def test_list_detection(self):
        assert DocumentChunker._has_list(
            "Features:\n- Reward points\n- Lounge access\n- Fuel surcharge waiver"
        )


# ═══════════════════════════════════════════════════════════════
#  PARSER TESTS
# ═══════════════════════════════════════════════════════════════

class TestDocumentParser:
    def test_validate_unsupported(self):
        valid, msg = DocumentParser.validate_file("test.xlsx")
        assert not valid
        assert "Unsupported" in msg

    def test_validate_not_found(self):
        valid, msg = DocumentParser.validate_file("/nonexistent/file.pdf")
        assert not valid
        assert "not found" in msg

    def test_parse_raw_text(self):
        docs = DocumentParser.parse_raw_text(
            "HDFC Bank credit card annual fee is Rs 10,000 for Infinia.",
            title="Test Content"
        )
        assert len(docs) == 1
        assert "Infinia" in docs[0].page_content
        assert docs[0].metadata["source_file"] == "Test Content"

    def test_clean_text(self):
        dirty = "Hello\n\n\n\n\nWorld   with    spaces"
        clean = DocumentParser._clean_text(dirty)
        assert "\n\n\n" not in clean
        assert "   " not in clean

    def test_get_file_info(self):
        # Create a temp file for testing
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode='w') as f:
            f.write("test content")
            path = f.name

        info = DocumentParser.get_file_info(path)
        assert info["file_type"] == "txt"
        assert info["file_size_bytes"] > 0

        import os
        os.unlink(path)


# ═══════════════════════════════════════════════════════════════
#  INTEGRATION TEST PLACEHOLDER
# ═══════════════════════════════════════════════════════════════

class TestRAGPipelineIntegration:
    """
    Integration tests — require PostgreSQL + ChromaDB running.
    Mark with @pytest.mark.integration and skip in CI without services.
    """

    @pytest.mark.skip(reason="Requires running PostgreSQL and ChromaDB")
    def test_full_indexing_pipeline(self):
        """Test: upload → parse → chunk → embed → store"""
        pass

    @pytest.mark.skip(reason="Requires running PostgreSQL and ChromaDB")
    def test_full_query_pipeline(self):
        """Test: question → embed → retrieve → generate → response"""
        pass
