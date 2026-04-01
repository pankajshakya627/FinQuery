"""
FastAPI API Routes.
Maps to architecture: FastAPI Backend on Cloud Run
Endpoints: /ingest, /query, /health, /documents, /stats
"""

import os
import shutil
from datetime import datetime
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, Query
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.session import get_async_session, check_db_connection
from app.db.mongodb import get_mongodb, check_mongodb_connection
from app.db.mongo_models import DocumentMongo, ChunkMongo, QueryLogMongo, DocumentStatus
from app.db.models import Document, Chunk, QueryLog, DocumentStatusEnum
from app.models.schemas import (
    DocumentMetadata, DocumentResponse, DocumentListResponse,
    IndexingConfig, IndexingResult, QueryRequest, RAGResponse,
    HealthCheck, SystemStats, ChunkStrategy,
)
from app.rag.pipeline import rag_pipeline
from app.rag.vector_store import vector_store

import structlog

logger = structlog.get_logger(__name__)
settings = get_settings()


# ═══════════════════════════════════════════════════════════════
#  ROUTERS
# ═══════════════════════════════════════════════════════════════

health_router = APIRouter(tags=["Health"])
document_router = APIRouter(prefix="/api/documents", tags=["Documents"])
query_router = APIRouter(prefix="/api", tags=["Query"])
stats_router = APIRouter(prefix="/api/stats", tags=["Statistics"])


# ═══════════════════════════════════════════════════════════════
#  HEALTH
# ═══════════════════════════════════════════════════════════════

@health_router.get("/health", response_model=HealthCheck)
async def health_check(
    db: AsyncSession = Depends(get_async_session),
    mongo_db = Depends(get_mongodb)
):
    """System health check — verifies PostgreSQL, MongoDB & ChromaDB connectivity."""
    pg_ok = await check_db_connection()
    mongo_ok = await check_mongodb_connection()
    chroma_ok = vector_store.is_healthy()

    # Get counts based on primary DB
    doc_count = 0
    chunk_count = 0
    
    try:
        if settings.database_type == "mongodb":
            doc_count = await mongo_db.documents.count_documents({})
            chunk_count = await mongo_db.chunks.count_documents({})
        else:
            result = await db.execute(select(func.count(Document.id)))
            doc_count = result.scalar() or 0
            result = await db.execute(select(func.count(Chunk.id)))
            chunk_count = result.scalar() or 0
    except Exception:
        pass

    primary_db_ok = mongo_ok if settings.database_type == "mongodb" else pg_ok

    return HealthCheck(
        status="healthy" if (primary_db_ok and chroma_ok) else "degraded",
        postgres_connected=pg_ok,
        mongodb_connected=mongo_ok,
        chroma_connected=chroma_ok,
        total_documents=doc_count,
        total_chunks=chunk_count,
        embedding_model=settings.embedding_model,
        llm_model=settings.llm_model,
        database_type=settings.database_type,
    )


# ═══════════════════════════════════════════════════════════════
#  DOCUMENT INGESTION
# ═══════════════════════════════════════════════════════════════

@document_router.post("/upload", response_model=IndexingResult)
async def upload_document(
    file: UploadFile = File(...),
    title: str = Form(...),
    description: str = Form(None),
    tags: str = Form(""),  # Comma-separated
    chunk_strategy: str = Form("recursive"),
    chunk_size: int = Form(512),
    chunk_overlap: int = Form(50),
    db: AsyncSession = Depends(get_async_session),
    mongo_db = Depends(get_mongodb),
):
    """
    Upload and index a document.
    Endpoint: POST /api/documents/upload
    Pipeline: File → Parse → Chunk → Embed → Store
    """
    # Validate file type
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    ext = os.path.splitext(file.filename)[1].lower()
    from app.rag.parser import SUPPORTED_EXTENSIONS
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}")

    # Save uploaded file
    os.makedirs(settings.upload_dir, exist_ok=True)
    doc_id = str(uuid4())
    safe_filename = f"{doc_id}{ext}"
    filepath = os.path.join(settings.upload_dir, safe_filename)

    try:
        with open(filepath, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(500, f"Failed to save file: {str(e)}")

    file_size = os.path.getsize(filepath)
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

    # Build metadata
    doc_meta = DocumentMetadata(
        id=doc_id,
        title=title,
        filename=file.filename,
        file_type=ext.lstrip('.'),
        file_size_bytes=file_size,
        description=description,
        tags=tag_list,
    )

    # Build indexing config
    config = IndexingConfig(
        chunk_strategy=ChunkStrategy(chunk_strategy),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Run indexing pipeline
    if settings.database_type == "mongodb":
        result = await rag_pipeline.index_document(filepath, doc_meta, config)
    else:
        result = await rag_pipeline.index_document(filepath, doc_meta, config, db)

    if result.errors:
        logger.warning("indexing_had_errors", errors=result.errors)

    return result


@document_router.post("/index-text", response_model=IndexingResult)
async def index_text(
    title: str = Form(...),
    content: str = Form(...),
    chunk_size: int = Form(512),
    db: AsyncSession = Depends(get_async_session),
    mongo_db = Depends(get_mongodb),
):
    """Index raw text content directly without file upload."""
    config = IndexingConfig(chunk_size=chunk_size)
    if settings.database_type == "mongodb":
        result = await rag_pipeline.index_raw_text(content, title, config=config)
    else:
        result = await rag_pipeline.index_raw_text(content, title, db, config)
    return result


@document_router.get("/", response_model=DocumentListResponse)
async def list_documents(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_async_session),
    mongo_db = Depends(get_mongodb),
):
    """List all indexed documents with pagination."""
    if settings.database_type == "mongodb":
        filter_query = {}
        if status:
            filter_query["status"] = status
        
        total = await mongo_db.documents.count_documents(filter_query)
        cursor = mongo_db.documents.find(filter_query).sort("created_at", -1).skip((page - 1) * page_size).limit(page_size)
        docs = await cursor.to_list(length=page_size)
        
        return DocumentListResponse(
            documents=[
                DocumentResponse(
                    id=d["_id"], title=d["title"], filename=d["filename"],
                    file_type=d["file_type"], status=d["status"],
                    total_chunks=d.get("total_chunks", 0), page_count=d.get("page_count", 0),
                    tags=d.get("tags") or [], created_at=d["created_at"],
                    indexed_at=d.get("indexed_at"),
                )
                for d in docs
            ],
            total=total,
            page=page,
            page_size=page_size,
        )
    else:
        query = select(Document).order_by(desc(Document.created_at))
        if status:
            query = query.where(Document.status == status)

        count_q = select(func.count(Document.id))
        if status:
            count_q = count_q.where(Document.status == status)
        total = (await db.execute(count_q)).scalar() or 0

        query = query.offset((page - 1) * page_size).limit(page_size)
        result = await db.execute(query)
        docs = result.scalars().all()

        return DocumentListResponse(
            documents=[
                DocumentResponse(
                    id=d.id, title=d.title, filename=d.filename,
                    file_type=d.file_type, status=d.status.value,
                    total_chunks=d.total_chunks, page_count=d.page_count,
                    tags=d.tags or [], created_at=d.created_at,
                    indexed_at=d.indexed_at,
                )
                for d in docs
            ],
            total=total,
            page=page,
            page_size=page_size,
        )


@document_router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    db: AsyncSession = Depends(get_async_session),
    mongo_db = Depends(get_mongodb),
):
    """Get document details by ID."""
    if settings.database_type == "mongodb":
        doc = await mongo_db.documents.find_one({"_id": document_id})
        if not doc:
            raise HTTPException(404, "Document not found")
        return DocumentResponse(
            id=doc["_id"], title=doc["title"], filename=doc["filename"],
            file_type=doc["file_type"], status=doc["status"],
            total_chunks=doc.get("total_chunks", 0), page_count=doc.get("page_count", 0),
            tags=doc.get("tags") or [], created_at=doc["created_at"],
            indexed_at=doc.get("indexed_at"),
        )
    else:
        result = await db.execute(
            select(Document).where(Document.id == document_id)
        )
        doc = result.scalar_one_or_none()
        if not doc:
            raise HTTPException(404, "Document not found")

        return DocumentResponse(
            id=doc.id, title=doc.title, filename=doc.filename,
            file_type=doc.file_type, status=doc.status.value,
            total_chunks=doc.total_chunks, page_count=doc.page_count,
            tags=doc.tags or [], created_at=doc.created_at,
            indexed_at=doc.indexed_at,
        )


@document_router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    db: AsyncSession = Depends(get_async_session),
    mongo_db = Depends(get_mongodb),
):
    """Delete a document and all its chunks/embeddings."""
    if settings.database_type == "mongodb":
        deleted = await rag_pipeline.delete_document(document_id)
    else:
        deleted = await rag_pipeline.delete_document(document_id, db)
    
    if not deleted:
        raise HTTPException(404, "Document not found")
    return {"message": "Document deleted", "document_id": document_id}


# ═══════════════════════════════════════════════════════════════
#  QUERY (RAG)
# ═══════════════════════════════════════════════════════════════

@query_router.post("/query", response_model=RAGResponse)
async def query_documents(
    request: QueryRequest,
    db: AsyncSession = Depends(get_async_session),
    mongo_db = Depends(get_mongodb),
):
    """
    RAG Query endpoint.
    Pipeline: Question → Embed → Retrieve → Rerank → Generate → Response
    """
    if settings.database_type == "mongodb":
        response = await rag_pipeline.query(request)
    else:
        response = await rag_pipeline.query(request, db)
    return response


@query_router.get("/query/simple")
async def simple_query(
    q: str = Query(..., min_length=3, description="Question"),
    top_k: int = Query(5, ge=1, le=20),
    db: AsyncSession = Depends(get_async_session),
    mongo_db = Depends(get_mongodb),
):
    """Simple GET-based query for quick testing."""
    request = QueryRequest(question=q, top_k=top_k)
    if settings.database_type == "mongodb":
        response = await rag_pipeline.query(request)
    else:
        response = await rag_pipeline.query(request, db)
    return response


# ═══════════════════════════════════════════════════════════════
#  STATISTICS
# ═══════════════════════════════════════════════════════════════

@stats_router.get("/", response_model=SystemStats)
async def get_stats(
    db: AsyncSession = Depends(get_async_session),
    mongo_db = Depends(get_mongodb),
):
    """Get system-wide statistics."""
    if settings.database_type == "mongodb":
        doc_count = await mongo_db.documents.count_documents({})
        chunk_count = await mongo_db.chunks.count_documents({})
        query_count = await mongo_db.query_logs.count_documents({})
        
        # Average processing time
        pipeline = [
            {"$group": {"_id": None, "avg_time": {"$avg": "$processing_time_ms"}}}
        ]
        stats_res = await mongo_db.query_logs.aggregate(pipeline).to_list(length=1)
        avg_time = stats_res[0]["avg_time"] if stats_res else 0
    else:
        doc_count = (await db.execute(select(func.count(Document.id)))).scalar() or 0
        chunk_count = (await db.execute(select(func.count(Chunk.id)))).scalar() or 0
        query_count = (await db.execute(select(func.count(QueryLog.id)))).scalar() or 0

        avg_time = (await db.execute(
            select(func.avg(QueryLog.processing_time_ms))
        )).scalar() or 0

    return SystemStats(
        total_documents=doc_count,
        total_chunks=chunk_count,
        total_queries=query_count,
        avg_response_time_ms=round(avg_time, 2),
    )


@stats_router.get("/vector-store")
async def vector_store_stats():
    """Get ChromaDB vector store statistics."""
    return vector_store.get_collection_stats()


@stats_router.get("/recent-queries")
async def recent_queries(
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_async_session),
    mongo_db = Depends(get_mongodb),
):
    """Get recent query logs."""
    if settings.database_type == "mongodb":
        cursor = mongo_db.query_logs.find().sort("created_at", -1).limit(limit)
        queries = await cursor.to_list(length=limit)
        
        return [
            {
                "id": q["_id"],
                "question": q["question"],
                "model": q.get("model_used"),
                "chunks_retrieved": q.get("chunks_retrieved", 0),
                "avg_score": q.get("avg_relevance_score", 0),
                "time_ms": q.get("processing_time_ms", 0),
                "created_at": q["created_at"].isoformat() if q.get("created_at") else None,
            }
            for q in queries
        ]
    else:
        result = await db.execute(
            select(QueryLog)
            .order_by(desc(QueryLog.created_at))
            .limit(limit)
        )
        queries = result.scalars().all()

        return [
            {
                "id": q.id,
                "question": q.question,
                "model": q.model_used,
                "chunks_retrieved": q.chunks_retrieved,
                "avg_score": q.avg_relevance_score,
                "time_ms": q.processing_time_ms,
                "created_at": q.created_at.isoformat() if q.created_at else None,
            }
            for q in queries
        ]
