"""
Pydantic models for request/response validation.
"""
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4
from pydantic import BaseModel, Field, ConfigDict


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"

class ChunkStrategy(str, Enum):
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    SENTENCE = "sentence"


# ── Document Models ──

class DocumentUpload(BaseModel):
    title: str = Field(..., min_length=1, max_length=256)
    description: Optional[str] = Field(None, max_length=1024)
    source_url: Optional[str] = None
    tags: list[str] = Field(default_factory=list)

class DocumentMetadata(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    filename: str
    file_type: str
    file_size_bytes: int
    description: Optional[str] = None
    source_url: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    status: DocumentStatus = DocumentStatus.PENDING
    total_chunks: int = 0
    page_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    indexed_at: Optional[datetime] = None
    error_message: Optional[str] = None

class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    title: str
    filename: str
    file_type: str
    status: DocumentStatus
    total_chunks: int
    page_count: int
    tags: list[str]
    created_at: datetime
    indexed_at: Optional[datetime] = None

class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int
    page: int
    page_size: int


# ── Chunk Models ──

class ChunkRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    chunk_index: int
    content: str
    section_title: Optional[str] = None
    page_number: Optional[int] = None
    token_count: int = 0
    embedding_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ── Query & RAG Models ──

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2048)
    top_k: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    include_sources: bool = True
    conversation_id: Optional[str] = None

class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    document_title: str
    section_title: Optional[str] = None
    content: str
    relevance_score: float
    page_number: Optional[int] = None

class RetrievalStats(BaseModel):
    total_chunks_searched: int
    chunks_retrieved: int
    top_k_used: int
    avg_relevance_score: float
    max_relevance_score: float
    min_relevance_score: float

class RAGResponse(BaseModel):
    answer: str
    question: str
    sources: list[RetrievedChunk] = Field(default_factory=list)
    model_used: str
    retrieval_stats: RetrievalStats
    processing_time_ms: float
    conversation_id: Optional[str] = None


# ── Indexing Models ──

class IndexingConfig(BaseModel):
    chunk_strategy: ChunkStrategy = ChunkStrategy.RECURSIVE
    chunk_size: int = Field(default=512, ge=100, le=4096)
    chunk_overlap: int = Field(default=50, ge=0, le=512)
    extract_metadata: bool = True
    generate_summaries: bool = False

class IndexingResult(BaseModel):
    document_id: str
    status: DocumentStatus
    chunks_created: int
    embeddings_stored: int
    processing_time_ms: float
    errors: list[str] = Field(default_factory=list)


# ── Health ──

class HealthCheck(BaseModel):
    status: str = "healthy"
    postgres_connected: bool = False
    chroma_connected: bool = False
    total_documents: int = 0
    total_chunks: int = 0
    embedding_model: str = ""
    llm_model: str = ""
    uptime_seconds: float = 0.0

class SystemStats(BaseModel):
    total_documents: int = 0
    total_chunks: int = 0
    total_queries: int = 0
    avg_response_time_ms: float = 0.0
    most_queried_topics: list[str] = Field(default_factory=list)
