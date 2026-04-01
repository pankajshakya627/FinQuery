"""
SQLAlchemy ORM models for PostgreSQL.
Tables: documents, chunks, query_logs, conversations.
"""
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, Text, Boolean, DateTime,
    ForeignKey, JSON, Enum as SAEnum, Index
)
from sqlalchemy.orm import relationship, DeclarativeBase
from sqlalchemy.sql import func
import enum


class Base(DeclarativeBase):
    pass

class DocumentStatusEnum(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class Document(Base):
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True)
    title = Column(String(256), nullable=False, index=True)
    filename = Column(String(512), nullable=False)
    file_type = Column(String(20), nullable=False)
    file_size_bytes = Column(Integer, default=0)
    description = Column(Text, nullable=True)
    source_url = Column(String(2048), nullable=True)
    tags = Column(JSON, default=list)
    status = Column(
        SAEnum(DocumentStatusEnum, name="document_status"),
        default=DocumentStatusEnum.PENDING, nullable=False, index=True
    )
    total_chunks = Column(Integer, default=0)
    page_count = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    indexed_at = Column(DateTime, nullable=True)

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_documents_status_created", "status", "created_at"),
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(String(36), primary_key=True)
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    section_title = Column(String(512), nullable=True)
    page_number = Column(Integer, nullable=True)
    token_count = Column(Integer, default=0)
    char_count = Column(Integer, default=0)
    embedding_id = Column(String(36), nullable=True)
    has_table = Column(Boolean, default=False)
    has_list = Column(Boolean, default=False)
    keywords = Column(JSON, default=list)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index("idx_chunks_doc_index", "document_id", "chunk_index"),
    )


class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(String(36), primary_key=True)
    conversation_id = Column(String(36), nullable=True, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=True)
    model_used = Column(String(64), nullable=True)
    top_k = Column(Integer, default=5)
    chunks_retrieved = Column(Integer, default=0)
    avg_relevance_score = Column(Float, default=0.0)
    processing_time_ms = Column(Float, default=0.0)
    feedback_rating = Column(Integer, nullable=True)
    feedback_text = Column(Text, nullable=True)
    retrieved_chunk_ids = Column(JSON, default=list)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("idx_query_logs_created", "created_at"),
    )


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True)
    title = Column(String(256), nullable=True)
    messages = Column(JSON, default=list)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
