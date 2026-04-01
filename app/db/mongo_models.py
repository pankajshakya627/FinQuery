"""
MongoDB Storage Models (Pydantic).
Used for document metadata, chunks, and logs in MongoDB.
"""
from datetime import datetime
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum
import uuid

class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"

class MongoBaseModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)

class DocumentMongo(MongoBaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), alias="_id")
    title: str
    filename: str
    file_type: str
    file_size_bytes: int = 0
    description: Optional[str] = None
    source_url: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    status: DocumentStatus = DocumentStatus.PENDING
    total_chunks: int = 0
    page_count: int = 0
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    indexed_at: Optional[datetime] = None

class ChunkMongo(MongoBaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), alias="_id")
    document_id: str
    chunk_index: int
    content: str
    section_title: Optional[str] = None
    page_number: Optional[int] = None
    token_count: int = 0
    char_count: int = 0
    embedding_id: Optional[str] = None
    has_table: bool = False
    has_list: bool = False
    keywords: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class QueryLogMongo(MongoBaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), alias="_id")
    conversation_id: Optional[str] = None
    question: str
    answer: Optional[str] = None
    model_used: Optional[str] = None
    top_k: int = 5
    chunks_retrieved: int = 0
    avg_relevance_score: float = 0.0
    processing_time_ms: float = 0.0
    feedback_rating: Optional[int] = None
    feedback_text: Optional[str] = None
    retrieved_chunk_ids: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ConversationMongo(MongoBaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), alias="_id")
    title: Optional[str] = None
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

# Utility function to convert SQLAlchemy models to Mongo-friendly dicts
def to_mongo_dict(obj: Any) -> Dict[str, Any]:
    """Convert an object or Pydantic model to a MongoDB-compatible dictionary."""
    if hasattr(obj, "dict"):
        data = obj.dict(by_alias=True)
    elif hasattr(obj, "model_dump"):
        data = obj.model_dump(by_alias=True)
    else:
        data = dict(obj)
    
    # Ensure ID is handled correctly
    if "id" in data and "_id" not in data:
        data["_id"] = data.pop("id")
    return data
