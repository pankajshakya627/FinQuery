"""
Embedding + Vector Store Layer.
Maps to architecture:
  - Embedding Layer: Sentence Transformers (open-source / self-hosted)
  - Vector Store Layer: ChromaDB (local persistent)

Handles: embedding generation, storage, similarity search, and CRUD.
"""

import os
from typing import Optional
from uuid import uuid4

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain.schema import Document as LCDocument
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_chroma import Chroma

from app.core.config import get_settings

import structlog

logger = structlog.get_logger(__name__)
settings = get_settings()


class VectorStoreManager:
    """
    Manages ChromaDB vector store with Sentence Transformer embeddings.
    Provides: indexing, retrieval, deletion, and collection management.
    """

    def __init__(self):
        self._embedding_fn = None
        self._chroma_client = None
        self._collection = None
        self._langchain_store = None

    # ── Initialization ──

    def initialize(self):
        """Initialize embedding model and ChromaDB."""
        logger.info("initializing_vector_store",
                     model=settings.embedding_model,
                     persist_dir=settings.chroma_persist_dir)

        # Ensure persist directory exists
        os.makedirs(settings.chroma_persist_dir, exist_ok=True)

        # Initialize embedding function (Sentence Transformers)
        self._embedding_fn = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 32,
            },
        )

        # Initialize ChromaDB persistent client
        self._chroma_client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Create LangChain-compatible Chroma wrapper
        self._langchain_store = Chroma(
            client=self._chroma_client,
            collection_name=settings.chroma_collection_name,
            embedding_function=self._embedding_fn,
        )

        # Get raw collection reference
        self._collection = self._chroma_client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info("vector_store_ready",
                     collection=settings.chroma_collection_name,
                     existing_docs=self._collection.count())

    # ── Indexing ──

    def index_chunks(
        self,
        chunks: list[LCDocument],
        document_id: str,
    ) -> list[str]:
        """
        Embed and store document chunks in ChromaDB.

        Args:
            chunks: LangChain Document objects with metadata
            document_id: Parent document ID for filtering

        Returns:
            List of ChromaDB IDs for stored embeddings
        """
        if not chunks:
            return []

        # ── Strip None/complex values from metadata (ChromaDB only accepts str/int/float/bool) ──
        chunks = filter_complex_metadata(chunks)

        # Prepare data for ChromaDB
        texts = [chunk.page_content for chunk in chunks]
        metadatas = []
        ids = []

        for chunk in chunks:
            chunk_id = chunk.metadata.get("chunk_id", str(uuid4()))
            ids.append(chunk_id)
            metadatas.append({
                "document_id": str(document_id),
                "chunk_index": int(chunk.metadata.get("chunk_index") or 0),
                "section_title": str(chunk.metadata.get("section_title") or ""),
                "page_number": int(chunk.metadata.get("page_number") or 0),
                "source_file": str(chunk.metadata.get("source_file") or ""),
                "token_count": int(chunk.metadata.get("token_count") or 0),
                "char_count": int(chunk.metadata.get("char_count") or 0),
            })

        # Store in ChromaDB via LangChain wrapper
        self._langchain_store.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
        )

        logger.info("chunks_indexed",
                     document_id=document_id,
                     chunks_stored=len(ids))

        return ids

    # ── Retrieval ──

    def similarity_search(
        self,
        query: str,
        top_k: int = None,
        score_threshold: float = None,
        filter_document_id: Optional[str] = None,
    ) -> list[tuple[LCDocument, float]]:
        """
        Retrieve Top-K most similar chunks for a query.
        Maps to: Query Processing → Top-K Semantic Retrieval

        Args:
            query: User question
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0-1)
            filter_document_id: Optional filter by specific document

        Returns:
            List of (Document, score) tuples, sorted by relevance
        """
        top_k = top_k or settings.top_k_results
        score_threshold = score_threshold or settings.similarity_threshold

        # Build filter
        where_filter = None
        if filter_document_id:
            where_filter = {"document_id": filter_document_id}

        # Perform similarity search with scores
        results = self._langchain_store.similarity_search_with_relevance_scores(
            query=query,
            k=top_k,
            score_threshold=score_threshold,
            filter=where_filter,
        )

        logger.info("similarity_search",
                     query=query[:80],
                     results_found=len(results),
                     top_k=top_k)

        return results

    # ── Management ──

    def delete_document_chunks(self, document_id: str) -> int:
        """Delete all chunks for a document from ChromaDB."""
        try:
            # Get all IDs for this document
            results = self._collection.get(
                where={"document_id": document_id},
            )
            if results["ids"]:
                self._collection.delete(ids=results["ids"])
                logger.info("chunks_deleted",
                           document_id=document_id,
                           count=len(results["ids"]))
                return len(results["ids"])
            return 0
        except Exception as e:
            logger.error("delete_failed", document_id=document_id, error=str(e))
            return 0

    def get_collection_stats(self) -> dict:
        """Get vector store statistics."""
        try:
            count = self._collection.count() if self._collection else 0
            return {
                "collection_name": settings.chroma_collection_name,
                "total_embeddings": count,
                "embedding_model": settings.embedding_model,
                "embedding_dimension": settings.embedding_dimension,
                "persist_directory": settings.chroma_persist_dir,
            }
        except Exception:
            return {"error": "Collection not initialized"}

    def is_healthy(self) -> bool:
        """Check if ChromaDB is operational."""
        try:
            self._collection.count()
            return True
        except Exception:
            return False

    def reset_collection(self):
        """Delete and recreate the collection."""
        try:
            self._chroma_client.delete_collection(settings.chroma_collection_name)
            self._collection = self._chroma_client.get_or_create_collection(
                name=settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._langchain_store = Chroma(
                client=self._chroma_client,
                collection_name=settings.chroma_collection_name,
                embedding_function=self._embedding_fn,
            )
            logger.info("collection_reset", collection=settings.chroma_collection_name)
        except Exception as e:
            logger.error("reset_failed", error=str(e))
            raise


# ── Singleton Instance ──
vector_store = VectorStoreManager()
