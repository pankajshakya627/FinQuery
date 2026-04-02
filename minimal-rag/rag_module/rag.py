#!/usr/bin/env python3
"""
Minimal RAG Engine with Ollama + Azure OpenAI support
PDFPlumber extraction + Table-aware chunking + Hybrid Search (Semantic + BM25 with RRF)
"""
import re
from typing import List
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader, TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from rag_module.config import get_settings
from rag_module.hybrid_retriever import HybridRetriever


class MinimalRAG:
    def __init__(self):
        self.settings = get_settings()
        self._embeddings = None
        self._vector_store = None
        self._llm = None
        self._chain = None
        self._documents = []

    def _init_embeddings(self):
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.settings.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
        return self._embeddings

    def _init_vector_store(self):
        if self._vector_store is None:
            self._vector_store = Chroma(
                collection_name=self.settings.chroma_collection_name,
                embedding_function=self._init_embeddings(),
                persist_directory=self.settings.chroma_persist_dir
            )
        return self._vector_store

    def _init_llm(self):
        if self._llm is None:
            provider = self.settings.llm_provider.lower()
            
            if provider == "ollama":
                self._llm = ChatOllama(
                    model=self.settings.llm_model,
                    temperature=self.settings.llm_temperature,
                    base_url=self.settings.ollama_base_url
                )
            elif provider == "azure":
                self._llm = AzureChatOpenAI(
                    azure_deployment=self.settings.azure_openai_ad_deployment_name,
                    openai_api_version=self.settings.azure_openai_api_version,
                    azure_endpoint=self.settings.azure_openai_endpoint,
                    api_key=self.settings.azure_openai_api_key,
                    temperature=self.settings.llm_temperature,
                    max_tokens=self.settings.llm_max_tokens
                )
            else:
                self._llm = ChatOpenAI(
                    model=self.settings.llm_model,
                    temperature=self.settings.llm_temperature,
                    max_tokens=self.settings.llm_max_tokens
                )
        return self._llm

    def _extract_tables_as_markdown(self, page) -> str:
        """Extract tables from a PDF page and convert to markdown."""
        tables = page.extract_tables()
        if not tables:
            return ""
        
        md_tables = []
        for i, table in enumerate(tables):
            if not table or len(table) < 2:
                continue
            
            # Convert to markdown table
            header = table[0]
            rows = table[1:]
            
            # Clean header
            header = [str(h).strip() if h else "" for h in header]
            md = "| " + " | ".join(header) + " |\n"
            md += "| " + " | ".join(["---"] * len(header)) + " |\n"
            
            for row in rows:
                row = [str(c).strip() if c else "" for c in row]
                # Pad row if needed
                while len(row) < len(header):
                    row.append("")
                md += "| " + " | ".join(row) + " |\n"
            
            md_tables.append(md)
        
        return "\n\n".join(md_tables)

    def _extract_page_content_with_tables(self, page, page_num: int) -> str:
        """Extract text and tables from a PDF page, preserving layout."""
        import pdfplumber
        
        # Get text with layout preservation
        text = page.extract_text() or ""
        
        # Get tables as markdown
        tables_md = self._extract_tables_as_markdown(page)
        
        # Combine text and tables
        content = text
        if tables_md:
            content += f"\n\n## Tables on Page {page_num + 1}\n\n" + tables_md
        
        return content

    def _load_pdf_with_tables(self, file_path: str) -> List[Document]:
        """Load PDF using pdfplumber with table extraction."""
        import pdfplumber
        import os
        
        docs = []
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                content = self._extract_page_content_with_tables(page, page_num)
                
                if content.strip():
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source_file": os.path.basename(file_path),
                            "file_type": "pdf",
                            "page_number": page_num + 1,
                            "total_pages": len(pdf.pages),
                            "has_tables": bool(page.extract_tables()),
                        }
                    )
                    docs.append(doc)
        
        return docs

    def load_and_chunk_document(self, file_path: str):
        """Load document with table-aware extraction and chunking."""
        import os
        from pathlib import Path
        
        path = Path(file_path)
        ext = path.suffix.lower()
        
        # Load document using appropriate loader
        if ext == ".pdf":
            docs = self._load_pdf_with_tables(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_file"] = os.path.basename(file_path)
                doc.metadata["file_type"] = "txt"
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        # Markdown header split for structure preservation
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "H1"), ("##", "H2"), ("###", "H3"), ("####", "H4"),
            ],
            strip_headers=False
        )
        
        header_splits = []
        for doc in docs:
            splits = md_splitter.split_text(doc.page_content)
            for split in splits:
                split.metadata = {**doc.metadata, **split.metadata}
                header_splits.append(split)
        
        section_docs = [
            Document(page_content=s.page_content, metadata=s.metadata)
            for s in header_splits
        ]
        
        # Table-aware recursive splitting
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            length_function=len,
            separators=[
                r"\n\n---\n\n",     # Table separator
                r"\n\n## ",         # H2 header
                r"\n\n# ",          # H1 header
                r"\n\n\n",          # Triple newline
                r"\n\n",            # Double newline
                r"\n",              # Single newline
                r"(?<=\.)\s+",      # Sentence boundary
                r" ",               # Word
            ],
            is_separator_regex=True,
        )
        
        chunks = splitter.split_documents(section_docs)
        
        # Enrich chunks with context
        for chunk in chunks:
            chunk.metadata["has_table"] = "|" in chunk.page_content
            chunk.metadata["char_count"] = len(chunk.page_content)
        
        self._documents.extend(chunks)
        
        store = self._init_vector_store()
        store.add_documents(chunks)
        
        return len(chunks)

    def add_document(self, text: str, metadata: dict = None):
        if metadata is None:
            metadata = {}
        
        doc = Document(page_content=text, metadata=metadata)
        
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "H1"), ("##", "H2"), ("###", "H3"),
            ],
            strip_headers=False
        )
        splits = md_splitter.split_text(text)
        
        if len(splits) > 1:
            docs = [Document(page_content=s.page_content, metadata=metadata) for s in splits]
        else:
            docs = [doc]
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            separators=[
                r"\n\n## ", r"\n\n# ", r"\n\n\n", r"\n\n", r"\n",
                r"(?<=\.)\s+", r" ",
            ],
            is_separator_regex=True,
        )
        
        chunks = splitter.split_documents(docs)
        self._documents.extend(chunks)
        
        store = self._init_vector_store()
        store.add_documents(chunks)
        return len(chunks)

    def _get_hybrid_retriever(self):
        """Get hybrid retriever with semantic + BM25 + RRF."""
        store = self._init_vector_store()
        
        collection = store._collection
        all_docs = collection.get(include=["documents", "metadatas"])
        
        documents = [
            Document(page_content=doc, metadata=meta or {})
            for doc, meta in zip(
                all_docs.get("documents", []) or [],
                all_docs.get("metadatas", []) or []
            )
        ]
        
        return HybridRetriever(
            vector_store=store,
            documents=documents,
            k=self.settings.top_k_results,
            rrf_k=60,
            vector_weight=0.5,
        )

    def _build_chain(self):
        if self._chain is None:
            retriever = self._get_hybrid_retriever()

            prompt = ChatPromptTemplate.from_template("""
You are a helpful financial assistant. Use the following context to answer the user's question.

Rules:
- If the context contains tables, preserve the table formatting in your answer
- Be specific with card names, fees, and charges
- If you don't know the answer, say exactly "I don't have information about that"
- Present information in a clear, structured format

Context:
{context}

Question: {question}

Answer:
            """)

            self._chain = (
                {"context": retriever | self._format_docs, "question": RunnablePassthrough()}
                | prompt
                | self._init_llm()
                | StrOutputParser()
            )
        return self._chain

    def _format_docs(self, docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    def query(self, question: str) -> str:
        chain = self._build_chain()
        return chain.invoke(question)
