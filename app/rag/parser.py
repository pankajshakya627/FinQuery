"""
Document Parser: Extracts text content from uploaded files.
Supports PDF, DOCX, TXT, HTML, and Markdown.
Maps to architecture: Sources & Ingestion → Parsing & Cleaning
"""

import os
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document as LCDocument

import structlog

logger = structlog.get_logger(__name__)


# ── Supported file types ──
SUPPORTED_EXTENSIONS = {
    ".pdf", ".txt", ".html", ".htm", ".md", ".csv",
    ".docx", ".doc", ".pptx", ".xlsx", ".xls", ".json", ".xml"
}


class DocumentParser:
    """
    Parsing & Cleaning Layer.
    Extracts text + metadata from various document formats.
    """

    @staticmethod
    def validate_file(filepath: str) -> tuple[bool, str]:
        """Validate file exists and is supported."""
        path = Path(filepath)
        if not path.exists():
            return False, f"File not found: {filepath}"
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return False, f"Unsupported file type: {path.suffix}. Supported: {SUPPORTED_EXTENSIONS}"
        max_size = 100 * 1024 * 1024  # Increased to 100 MB for spreadsheets/presentations
        if path.stat().st_size > max_size:
            return False, f"File too large: {path.stat().st_size / 1024 / 1024:.1f} MB (max: 100 MB)"
        return True, "OK"

    @staticmethod
    def parse(filepath: str) -> list[LCDocument]:
        """
        Parse a document file and return LangChain Document objects.
        Each page/section becomes a separate Document with metadata.
        """
        path = Path(filepath)
        ext = path.suffix.lower()

        logger.info("parsing_document", filepath=filepath, extension=ext)

        try:
            if ext == ".pdf":
                return DocumentParser._parse_pdf(filepath)
            elif ext == ".txt":
                return DocumentParser._parse_text(filepath)
            elif ext in (".html", ".htm"):
                return DocumentParser._parse_html(filepath)
            elif ext == ".md":
                return DocumentParser._parse_markdown(filepath)
            elif ext in (".docx", ".doc"):
                return DocumentParser._parse_docx(filepath)
            elif ext == ".pptx":
                return DocumentParser._parse_pptx(filepath)
            elif ext in (".xlsx", ".xls"):
                return DocumentParser._parse_excel(filepath)
            elif ext == ".csv":
                return DocumentParser._parse_csv(filepath)
            elif ext == ".json":
                return DocumentParser._parse_json(filepath)
            else:
                # Catch-all using Unstructured for other supported extensions
                return DocumentParser._parse_unstructured(filepath)
        except Exception as e:
            logger.error("parse_failed", filepath=filepath, error=str(e))
            raise

    @staticmethod
    def _parse_pdf(filepath: str) -> list[LCDocument]:
        """Parse PDF using IBM Docling for high-fidelity layout and table extraction."""
        from docling.document_converter import DocumentConverter
        import time

        logger.info("docling_conversion_started", filepath=filepath)
        start_time = time.time()
        
        converter = DocumentConverter()
        result = converter.convert(filepath)
        
        # Expert the entire parsed layout to structural Markdown
        md_text = result.document.export_to_markdown()
        
        doc = LCDocument(
            page_content=md_text,
            metadata={
                "source_file": os.path.basename(filepath),
                "file_type": "pdf",
                "parsed_by": "docling"
            }
        )

        logger.info("docling_conversion_complete", 
                    filepath=filepath, 
                    time_seconds=round(time.time() - start_time, 2),
                    char_count=len(md_text))
        
        return [doc]

    @staticmethod
    def _parse_text(filepath: str) -> list[LCDocument]:
        """Parse plain text file."""
        loader = TextLoader(filepath, encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.metadata.update({
                "source_file": os.path.basename(filepath),
                "file_type": "txt",
                "page_number": 1,
            })
            doc.page_content = DocumentParser._clean_text(doc.page_content)
        return docs

    @staticmethod
    def _parse_docx(filepath: str) -> list[LCDocument]:
        """Parse DOCX file."""
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(filepath)
        docs = loader.load()
        for doc in docs:
            doc.metadata.update({
                "source_file": os.path.basename(filepath),
                "file_type": "docx",
            })
        return docs

    @staticmethod
    def _parse_pptx(filepath: str) -> list[LCDocument]:
        """Parse PPTX file."""
        from langchain_community.document_loaders import UnstructuredPowerPointLoader
        loader = UnstructuredPowerPointLoader(filepath)
        docs = loader.load()
        for doc in docs:
            doc.metadata.update({
                "source_file": os.path.basename(filepath),
                "file_type": "pptx",
            })
        return docs

    @staticmethod
    def _parse_excel(filepath: str) -> list[LCDocument]:
        """Parse Excel file."""
        from langchain_community.document_loaders import UnstructuredExcelLoader
        loader = UnstructuredExcelLoader(filepath, mode="elements")
        docs = loader.load()
        for doc in docs:
            doc.metadata.update({
                "source_file": os.path.basename(filepath),
                "file_type": "xlsx",
            })
        return docs

    @staticmethod
    def _parse_csv(filepath: str) -> list[LCDocument]:
        """Parse CSV file."""
        from langchain_community.document_loaders.csv_loader import CSVLoader
        loader = CSVLoader(filepath)
        docs = loader.load()
        for doc in docs:
            doc.metadata.update({
                "source_file": os.path.basename(filepath),
                "file_type": "csv",
            })
        return docs

    @staticmethod
    def _parse_json(filepath: str) -> list[LCDocument]:
        """Parse JSON file."""
        from langchain_community.document_loaders import JSONLoader
        # Simple JSON loader that treats the whole file as one document by default or per key
        loader = JSONLoader(
            file_path=filepath,
            jq_schema=".",
            text_content=False
        )
        docs = loader.load()
        for doc in docs:
            doc.metadata.update({
                "source_file": os.path.basename(filepath),
                "file_type": "json",
            })
        return docs

    @staticmethod
    def _parse_unstructured(filepath: str) -> list[LCDocument]:
        """Parse using generic Unstructured loader."""
        from langchain_community.document_loaders import UnstructuredFileLoader
        loader = UnstructuredFileLoader(filepath)
        docs = loader.load()
        for doc in docs:
            doc.metadata.update({
                "source_file": os.path.basename(filepath),
                "file_type": Path(filepath).suffix.lower().lstrip('.'),
            })
        return docs

    @staticmethod
    def _parse_html(filepath: str) -> list[LCDocument]:
        """Parse HTML file."""
        loader = UnstructuredHTMLLoader(filepath)
        docs = loader.load()
        for doc in docs:
            doc.metadata.update({
                "source_file": os.path.basename(filepath),
                "file_type": "html",
            })
            doc.page_content = DocumentParser._clean_text(doc.page_content)
        return docs

    @staticmethod
    def _parse_markdown(filepath: str) -> list[LCDocument]:
        """Parse Markdown file."""
        loader = UnstructuredMarkdownLoader(filepath)
        docs = loader.load()
        for doc in docs:
            doc.metadata.update({
                "source_file": os.path.basename(filepath),
                "file_type": "md",
            })
        return docs

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean extracted text: remove excessive whitespace, fix encoding."""
        import re
        # Normalize whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)
        # Fix common PDF artifacts
        text = text.replace('\x00', '')
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)  # Fix word breaks
        return text.strip()

    @staticmethod
    def parse_raw_text(text: str, title: str = "Raw Text") -> list[LCDocument]:
        """Parse raw text string directly (for API/inline content)."""
        doc = LCDocument(
            page_content=DocumentParser._clean_text(text),
            metadata={
                "source_file": title,
                "file_type": "raw",
                "page_number": 1,
            }
        )
        return [doc]

    @staticmethod
    def get_file_info(filepath: str) -> dict:
        """Get file metadata without parsing content."""
        path = Path(filepath)
        return {
            "filename": path.name,
            "file_type": path.suffix.lower().lstrip('.'),
            "file_size_bytes": path.stat().st_size,
            "file_size_mb": round(path.stat().st_size / (1024 * 1024), 2),
        }
