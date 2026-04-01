"""
Chunking Layer: Splits parsed documents into semantically meaningful chunks.
Maps to architecture: Indexing Pipeline → Chunking Layer
Strategies: Recursive, Sentence-based, Semantic (LLM-guided).
"""

import re
from typing import Optional
from uuid import uuid4

from langchain.schema import Document as LCDocument
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

from app.core.config import get_settings
from app.models.schemas import ChunkStrategy

import structlog

logger = structlog.get_logger(__name__)
settings = get_settings()


class DocumentChunker:
    """
    Chunking Layer — splits documents into retrieval-optimized chunks.
    Output: list of chunks with metadata (section title, page, keywords).
    """

    def __init__(
        self,
        strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        self.strategy = strategy
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self._splitter = self._create_splitter()

    def _create_splitter(self) -> RecursiveCharacterTextSplitter:
        """Create the appropriate text splitter."""
        if self.strategy == ChunkStrategy.RECURSIVE:
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=[
                    "\n\n\n",     # Triple newline (section breaks)
                    "\n\n",       # Double newline (paragraphs)
                    "\n",         # Single newline
                    ". ",         # Sentence boundary
                    "; ",         # Clause boundary
                    ", ",         # Comma
                    " ",          # Word
                ],
                is_separator_regex=False,
            )
        elif self.strategy == ChunkStrategy.SENTENCE:
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", "? ", "! "],
            )
        else:
            # Default to recursive
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

    def chunk_documents(
        self,
        documents: list[LCDocument],
        document_id: str,
    ) -> list[LCDocument]:
        """
        Split documents into chunks with enriched metadata.

        Args:
            documents: Parsed LangChain Document objects (pages)
            document_id: Parent document ID for tracking

        Returns:
            List of chunked LCDocument objects with metadata
        """
        all_chunks = []
        chunk_index = 0

        for doc in documents:
            # Split this page/section
            raw_chunks = self._splitter.split_documents([doc])

            for chunk in raw_chunks:
                # Enrich metadata
                chunk.metadata.update({
                    "chunk_id": str(uuid4()),
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "section_title": self._extract_section_title(chunk.page_content),
                    "token_count": self._estimate_tokens(chunk.page_content),
                    "char_count": len(chunk.page_content),
                    "has_table": self._has_table(chunk.page_content),
                    "has_list": self._has_list(chunk.page_content),
                    "keywords": self._extract_keywords(chunk.page_content),
                })

                all_chunks.append(chunk)
                chunk_index += 1

        logger.info(
            "chunking_complete",
            document_id=document_id,
            strategy=self.strategy.value,
            total_chunks=len(all_chunks),
            avg_chunk_size=sum(len(c.page_content) for c in all_chunks) // max(len(all_chunks), 1),
        )

        return all_chunks

    @staticmethod
    def _extract_section_title(text: str) -> Optional[str]:
        """Extract section heading from chunk text."""
        lines = text.strip().split('\n')
        first_line = lines[0].strip() if lines else ""

        # Check if first line looks like a heading
        # Pattern: numbered section, all-caps, or short bold-style text
        heading_patterns = [
            r'^\d+\.\s+\w+',           # "1. Fees and Charges"
            r'^[A-Z][A-Z\s&/]+$',       # "SCHEDULE OF CHARGES"
            r'^[A-Z]\)\s+',             # "A) The Cardmember..."
            r'^#{1,3}\s+',              # Markdown headings
        ]

        for pattern in heading_patterns:
            if re.match(pattern, first_line) and len(first_line) < 100:
                return first_line.strip('#').strip()

        return None

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token count (~4 chars per token for English)."""
        return len(text) // 4

    @staticmethod
    def _has_table(text: str) -> bool:
        """Detect if chunk contains tabular data."""
        table_indicators = ['|', '\t\t', '  —  ', '---', 'Rs.', '₹']
        lines = text.split('\n')
        table_lines = sum(1 for line in lines if any(ind in line for ind in table_indicators))
        return table_lines >= 3

    @staticmethod
    def _has_list(text: str) -> bool:
        """Detect if chunk contains list items."""
        list_patterns = [r'^\s*[-•●]\s', r'^\s*\d+[.)]\s', r'^\s*[a-z][.)]\s']
        lines = text.split('\n')
        list_lines = sum(
            1 for line in lines
            if any(re.match(p, line) for p in list_patterns)
        )
        return list_lines >= 2

    @staticmethod
    def _extract_keywords(text: str, max_keywords: int = 10) -> list[str]:
        """Extract important keywords from chunk using frequency analysis."""
        # Remove common words
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'of', 'in', 'to', 'for',
            'with', 'on', 'at', 'by', 'from', 'and', 'or', 'not', 'no', 'but',
            'if', 'then', 'than', 'that', 'this', 'which', 'who', 'whom',
            'their', 'its', 'his', 'her', 'our', 'your', 'my', 'all', 'any',
            'each', 'every', 'such', 'same', 'other', 'also', 'as', 'so',
        }

        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        word_freq = {}
        for w in words:
            if w not in stop_words:
                word_freq[w] = word_freq.get(w, 0) + 1

        # Sort by frequency, take top N
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in sorted_words[:max_keywords]]
