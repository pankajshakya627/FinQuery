"""
RAG Pipeline Orchestrator.
Maps to the complete architecture flow:

  Ingestion:  Document → Parse → Chunk → Embed → Store (ChromaDB + PostgreSQL)
  Query:      Question → Embed → Retrieve → Rerank → Generate → Response

This is the central coordinator that connects all RAG components.
"""

import time
from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.mongodb import MongoDB
from app.db.mongo_models import DocumentMongo, ChunkMongo, QueryLogMongo, DocumentStatus, to_mongo_dict
from app.db.models import Document, Chunk, QueryLog, DocumentStatusEnum
from app.models.schemas import (
    DocumentMetadata, IndexingConfig, IndexingResult,
    QueryRequest, RAGResponse, RetrievedChunk, RetrievalStats,
    ChunkStrategy,
)
from app.rag.parser import DocumentParser
from app.rag.chunker import DocumentChunker
from app.rag.vector_store import vector_store
from app.rag.generator import answer_generator

import structlog

logger = structlog.get_logger(__name__)
settings = get_settings()


class RAGPipeline:
    """
    Complete RAG Pipeline orchestrating:
    1. Document ingestion & indexing
    2. Query processing & answer generation
    3. Conversation management
    """

    def __init__(self):
        self._initialized = False

    def initialize(self):
        """Initialize all RAG components."""
        if self._initialized:
            return

        logger.info("initializing_rag_pipeline")

        # Initialize vector store (embedding model + ChromaDB)
        vector_store.initialize()

        # Initialize LLM
        answer_generator.initialize()

        self._initialized = True
        logger.info("rag_pipeline_ready")

    # ═══════════════════════════════════════════════════════════
    #  INDEXING PIPELINE
    # ═══════════════════════════════════════════════════════════

    async def index_document(
        self,
        filepath: str,
        document_meta: DocumentMetadata,
        config: IndexingConfig,
        db: Optional[AsyncSession] = None,
    ) -> IndexingResult:
        """
        Full indexing pipeline:
        File → Parse → Chunk → Embed → Store in ChromaDB + (PostgreSQL or MongoDB)

        Args:
            filepath: Path to the uploaded file
            document_meta: Document metadata
            config: Indexing configuration
            db: PostgreSQL session (optional if using MongoDB)

        Returns:
            IndexingResult with stats
        """
        start_time = time.time()
        errors = []

        try:
            # ── Step 1: Initialize Database Record ──
            if settings.database_type == "mongodb":
                mongo_db = MongoDB.get_db()
                doc_record = DocumentMongo(
                    id=document_meta.id,
                    title=document_meta.title,
                    filename=document_meta.filename,
                    file_type=document_meta.file_type,
                    file_size_bytes=document_meta.file_size_bytes,
                    description=document_meta.description,
                    source_url=document_meta.source_url,
                    tags=document_meta.tags,
                    status=DocumentStatus.PROCESSING,
                )
                await mongo_db.documents.insert_one(to_mongo_dict(doc_record))
            else:
                doc_record = Document(
                    id=document_meta.id,
                    title=document_meta.title,
                    filename=document_meta.filename,
                    file_type=document_meta.file_type,
                    file_size_bytes=document_meta.file_size_bytes,
                    description=document_meta.description,
                    source_url=document_meta.source_url,
                    tags=document_meta.tags,
                    status=DocumentStatusEnum.PROCESSING,
                )
                db.add(doc_record)
                await db.flush()

            logger.info("indexing_started", document_id=document_meta.id,
                        title=document_meta.title, db_type=settings.database_type)

            # ── Step 2: Parse document ──
            parsed_docs = DocumentParser.parse(filepath)
            page_count = len(parsed_docs)

            if not parsed_docs:
                raise ValueError("No content extracted from document")

            # ── Step 3: Chunk documents ──
            chunker = DocumentChunker(
                strategy=config.chunk_strategy,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )
            chunks = chunker.chunk_documents(parsed_docs, document_meta.id)

            if not chunks:
                raise ValueError("No chunks generated from document")

            # ── Step 4: Store chunks in Database ──
            if settings.database_type == "mongodb":
                chunk_records = []
                for chunk in chunks:
                    chunk_records.append(to_mongo_dict(ChunkMongo(
                        id=chunk.metadata["chunk_id"],
                        document_id=document_meta.id,
                        chunk_index=chunk.metadata["chunk_index"],
                        content=chunk.page_content,
                        section_title=chunk.metadata.get("section_title"),
                        page_number=chunk.metadata.get("page_number"),
                        token_count=chunk.metadata.get("token_count", 0),
                        char_count=chunk.metadata.get("char_count", 0),
                        keywords=chunk.metadata.get("keywords", []),
                    )))
                if chunk_records:
                    await mongo_db.chunks.insert_many(chunk_records)
            else:
                for chunk in chunks:
                    chunk_record = Chunk(
                        id=chunk.metadata["chunk_id"],
                        document_id=document_meta.id,
                        chunk_index=chunk.metadata["chunk_index"],
                        content=chunk.page_content,
                        section_title=chunk.metadata.get("section_title"),
                        page_number=chunk.metadata.get("page_number"),
                        token_count=chunk.metadata.get("token_count", 0),
                        char_count=chunk.metadata.get("char_count", 0),
                        keywords=chunk.metadata.get("keywords", []),
                    )
                    db.add(chunk_record)

            # ── Step 5: Embed & store in ChromaDB ──
            embedding_ids = await vector_store.index_chunks_async(chunks, document_meta.id)

            # Update chunk records with embedding IDs
            for chunk, emb_id in zip(chunks, embedding_ids):
                chunk.metadata["embedding_id"] = emb_id

            # ── Step 6: Update document status ──
            processing_time = (time.time() - start_time) * 1000

            if settings.database_type == "mongodb":
                await mongo_db.documents.update_one(
                    {"_id": document_meta.id},
                    {
                        "$set": {
                            "status": DocumentStatus.INDEXED,
                            "total_chunks": len(chunks),
                            "page_count": page_count,
                            "indexed_at": datetime.utcnow(),
                            "updated_at": datetime.utcnow()
                        }
                    }
                )
            else:
                doc_record.status = DocumentStatusEnum.INDEXED
                doc_record.total_chunks = len(chunks)
                doc_record.page_count = page_count
                doc_record.indexed_at = datetime.utcnow()
                await db.commit()

            logger.info("indexing_complete",
                        document_id=document_meta.id,
                        chunks=len(chunks),
                        embeddings=len(embedding_ids),
                        time_ms=processing_time)

            return IndexingResult(
                document_id=document_meta.id,
                status=DocumentStatusEnum.INDEXED,
                chunks_created=len(chunks),
                embeddings_stored=len(embedding_ids),
                processing_time_ms=processing_time,
                errors=errors,
            )

        except Exception as e:
            logger.error("indexing_failed",
                        document_id=document_meta.id,
                        error=str(e))

            # Update status to FAILED
            if settings.database_type == "mongodb":
                mongo_db = mongo_db or MongoDB.get_db()
                if mongo_db is not None:
                    await mongo_db.documents.update_one(
                    {"_id": document_meta.id},
                    {
                        "$set": {
                            "status": DocumentStatus.FAILED,
                            "error_message": str(e),
                            "updated_at": datetime.utcnow()
                        }
                    }
                )
            else:
                try:
                    doc_record.status = DocumentStatusEnum.FAILED
                    doc_record.error_message = str(e)
                    await db.commit()
                except Exception:
                    if db: await db.rollback()

            processing_time = (time.time() - start_time) * 1000

            return IndexingResult(
                document_id=document_meta.id,
                status=DocumentStatusEnum.FAILED,
                chunks_created=0,
                embeddings_stored=0,
                processing_time_ms=processing_time,
                errors=[str(e)],
            )

    async def index_raw_text(
        self,
        text: str,
        title: str,
        db: AsyncSession,
        config: IndexingConfig = None,
    ) -> IndexingResult:
        """Index raw text content directly (no file upload needed)."""
        config = config or IndexingConfig()
        doc_id = str(uuid4())

        meta = DocumentMetadata(
            id=doc_id,
            title=title,
            filename=f"{title.lower().replace(' ', '_')}.txt",
            file_type="txt",
            file_size_bytes=len(text.encode()),
            description=f"Raw text: {title}",
        )

        # Parse raw text
        parsed = DocumentParser.parse_raw_text(text, title)

        # Write to temp file for the pipeline
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(text)
            temp_path = f.name

        try:
            result = await self.index_document(temp_path, meta, config, db)
        finally:
            os.unlink(temp_path)

        return result

    # ═══════════════════════════════════════════════════════════
    #  QUERY PIPELINE
    # ═══════════════════════════════════════════════════════════

    async def query(
        self,
        request: QueryRequest,
        db: Optional[AsyncSession] = None,
    ) -> RAGResponse:
        """
        Full RAG query pipeline:
        Question → Embed → Retrieve → Rerank → Generate → Response

        Args:
            request: User query with parameters
            db: Database session (optional if using MongoDB)

        Returns:
            RAGResponse with answer, sources, and stats
        """
        start_time = time.time()

        logger.info("query_started", question=request.question[:80], db_type=settings.database_type)

        # ── Step 1: Retrieve from vector store ──
        search_results = await vector_store.similarity_search_async(
            query=request.question,
            top_k=request.top_k,
            score_threshold=request.similarity_threshold,
        )
        logger.info("retrieval_complete", results_found=len(search_results))

        # ── Step 2: Build retrieved chunks with metadata ──
        retrieved_chunks = []
        context_docs = []

        logger.info("processing_chunks_metadata", count=len(search_results))
        for doc, score in search_results:
            # Look up document title from DB
            doc_title = await self._get_document_title(
                db, doc.metadata.get("document_id", "")
            )

            retrieved_chunks.append(RetrievedChunk(
                chunk_id=doc.metadata.get("chunk_id", ""),
                document_id=doc.metadata.get("document_id", ""),
                document_title=doc_title,
                section_title=doc.metadata.get("section_title"),
                content=doc.page_content,
                relevance_score=round(score, 4),
                page_number=doc.metadata.get("page_number"),
            ))
            context_docs.append(doc)

        # ── Step 3: Generate answer ──
        if context_docs:
            # Get conversation history if multi-turn
            conv_history = None
            if request.conversation_id:
                conv_history = await self._get_conversation_history(
                    db, request.conversation_id
                )

            logger.info("calling_generator", doc_count=len(context_docs))
            answer = await answer_generator.generate(
                question=request.question,
                context_docs=context_docs,
                conversation_history=conv_history,
            )
        else:
            answer = (
                "I couldn't find relevant information in the indexed documents "
                "for your question. Please try:\n"
                "- Rephrasing your question\n"
                "- Asking about specific credit card fees, charges, or policies\n"
                "- Ensuring documents have been indexed"
            )

        processing_time = (time.time() - start_time) * 1000

        # ── Step 4: Compute retrieval stats ──
        scores = [c.relevance_score for c in retrieved_chunks]
        total_chunks = await self._get_total_chunks(db)

        stats = RetrievalStats(
            total_chunks_searched=total_chunks,
            chunks_retrieved=len(retrieved_chunks),
            top_k_used=request.top_k,
            avg_relevance_score=round(sum(scores) / max(len(scores), 1), 4),
            max_relevance_score=round(max(scores) if scores else 0, 4),
            min_relevance_score=round(min(scores) if scores else 0, 4),
        )

        # ── Step 5: Log query ──
        if settings.database_type == "mongodb":
            mongo_db = MongoDB.get_db()
            query_log = QueryLogMongo(
                id=str(uuid4()),
                conversation_id=request.conversation_id,
                question=request.question,
                answer=answer,
                model_used=answer_generator.model_name,
                top_k=request.top_k,
                chunks_retrieved=len(retrieved_chunks),
                avg_relevance_score=stats.avg_relevance_score,
                processing_time_ms=processing_time,
                retrieved_chunk_ids=[c.chunk_id for c in retrieved_chunks],
            )
            await mongo_db.query_logs.insert_one(to_mongo_dict(query_log))
        else:
            query_log = QueryLog(
                id=str(uuid4()),
                conversation_id=request.conversation_id,
                question=request.question,
                answer=answer,
                model_used=answer_generator.model_name,
                top_k=request.top_k,
                chunks_retrieved=len(retrieved_chunks),
                avg_relevance_score=stats.avg_relevance_score,
                processing_time_ms=processing_time,
                retrieved_chunk_ids=[c.chunk_id for c in retrieved_chunks],
            )
            db.add(query_log)
            await db.commit()

        logger.info("query_complete",
                     question=request.question[:80],
                     chunks_retrieved=len(retrieved_chunks),
                     time_ms=processing_time)

        return RAGResponse(
            answer=answer,
            question=request.question,
            sources=retrieved_chunks if request.include_sources else [],
            model_used=answer_generator.model_name,
            retrieval_stats=stats,
            processing_time_ms=round(processing_time, 2),
            conversation_id=request.conversation_id,
        )

    # ═══════════════════════════════════════════════════════════
    #  HELPERS
    # ═══════════════════════════════════════════════════════════

    async def _get_document_title(self, db: Optional[AsyncSession], doc_id: str) -> str:
        """Look up document title from Database."""
        try:
            if settings.database_type == "mongodb":
                mongo_db = MongoDB.get_db()
                doc = await mongo_db.documents.find_one({"_id": doc_id}, {"title": 1})
                return doc["title"] if doc else "Unknown Document"
            else:
                result = await db.execute(
                    select(Document.title).where(Document.id == doc_id)
                )
                title = result.scalar_one_or_none()
                return title or "Unknown Document"
        except Exception:
            return "Unknown Document"

    async def _get_total_chunks(self, db: Optional[AsyncSession]) -> int:
        """Get total chunk count."""
        try:
            if settings.database_type == "mongodb":
                mongo_db = MongoDB.get_db()
                return await mongo_db.chunks.count_documents({})
            else:
                result = await db.execute(select(func.count(Chunk.id)))
                return result.scalar() or 0
        except Exception:
            return 0

    async def _get_conversation_history(
        self, db: Optional[AsyncSession], conversation_id: str
    ) -> list[dict]:
        """Get recent conversation messages."""
        try:
            if settings.database_type == "mongodb":
                mongo_db = MongoDB.get_db()
                conv = await mongo_db.conversations.find_one({"_id": conversation_id}, {"messages": 1})
                return conv["messages"] if conv else []
            else:
                from app.db.models import Conversation
                result = await db.execute(
                    select(Conversation.messages).where(
                        Conversation.id == conversation_id
                    )
                )
                messages = result.scalar_one_or_none()
                return messages or []
        except Exception:
            return []

    async def delete_document(self, document_id: str, db: Optional[AsyncSession] = None) -> bool:
        """Delete document and all associated data."""
        try:
            # Delete from ChromaDB
            vector_store.delete_document_chunks(document_id)

            # Delete from Database
            if settings.database_type == "mongodb":
                mongo_db = MongoDB.get_db()
                # Delete document
                doc_res = await mongo_db.documents.delete_one({"_id": document_id})
                # Delete chunks
                await mongo_db.chunks.delete_many({"document_id": document_id})
                
                if doc_res.deleted_count > 0:
                    logger.info("document_deleted_mongo", document_id=document_id)
                    return True
            else:
                # Delete from PostgreSQL (cascades to chunks)
                result = await db.execute(
                    select(Document).where(Document.id == document_id)
                )
                doc = result.scalar_one_or_none()
                if doc:
                    await db.delete(doc)
                    await db.commit()
                    logger.info("document_deleted_pg", document_id=document_id)
                    return True
            return False
        except Exception as e:
            logger.error("delete_failed", document_id=document_id, error=str(e))
            if db: await db.rollback()
            return False


# ── Singleton ──
rag_pipeline = RAGPipeline()
