# FinQuery — HDFC Credit Card RAG Q&A System

Production-grade Retrieval-Augmented Generation system for HDFC Bank Credit Card MITC (Most Important Terms & Conditions) document Q&A. Now with **MongoDB** metadata storage, **Ollama** local LLM support, and **Universal Document Parsing**.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    INDEXING PIPELINE                                        │
│                                                                             │
│  📄 Document Upload    →  🔍 Universal Parser  →  ✂️ LangChain Chunker       │
│  (PDF/Word/Excel/PPTX/  (Auto-format detection)  (Recursive / Metadata)     │
│   CSV/JSON/HTML/MD)                                                         │
│                                                                             │
│       →  🧮 Sentence Transformers    →  💾 ChromaDB + MongoDB                │
│          (all-MiniLM-L6-v2)              (Vector Store + Metadata)          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    ONLINE QUERY PATH                                        │
│                                                                             │
│  ❓ User Question   →  🧮 Query Embedding   →  🔎 Top-K Retrieval            │
│  (FastAPI endpoint)     (Same model)            (Cosine similarity)         │
│                                                                             │
│       →  📊 Reranking     →  🤖 LLM Generation    →  ✅ Grounded Answer      │
│          (Score filter)      (Ollama/OpenAI)          (+ Source Citations)  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Features

- **Universal Document Support**: Index PDF, docx, xlsx, pptx, csv, json, html, and markdown.
- **Database Agnostic**: Support for both **MongoDB** (Async via Motor) and **PostgreSQL** (SQLAlchemy).
- **Local LLM Support**: Fully integrated with **Ollama** (e.g., Qwen2.5, Llama3) for private, offline inference.
- **Dynamic Metadata**: Advanced metadata filtering for ChromaDB to support complex document structures.
- **Production Dashboard**: Real-time health monitoring, vector store statistics, and query logs.

## Tech Stack

| Layer              | Technology                         |
|--------------------|------------------------------------|
| **API Framework**  | FastAPI + Uvicorn                  |
| **Validation**     | Pydantic v2 + Pydantic Settings    |
| **RAG Framework**  | LangChain 0.3                      |
| **Embeddings**     | Sentence Transformers (MiniLM)     |
| **Vector Store**   | ChromaDB (persistent, local)       |
| **Metadata DB**    | **MongoDB 7.0** (or PostgreSQL 16) |
| **LLM**           | **Ollama** (Local), OpenAI, or **Azure OpenAI** |
| **Frontend**       | Jinja2 + Vanilla JS (Dark Theme)   |

## Project Structure

```
hdfc-rag/
├── app/
│   ├── api/
│   │   └── routes.py          # FastAPI endpoints (ingest, query, health)
│   ├── core/
│   │   └── config.py          # Configuration management
│   ├── db/
│   │   ├── mongodb.py         # Async MongoDB client management
│   │   ├── mongo_models.py    # Pydantic models for MongoDB
│   │   ├── models.py          # SQLAlchemy models (PostgreSQL)
│   │   └── session.py         # Lazy engine initialization
│   ├── rag/
│   │   ├── parser.py          # Universal parser (10+ formats)
│   │   ├── chunker.py         # Text splitting logic
│   │   ├── vector_store.py    # ChromaDB + metadata filtering
│   │   ├── generator.py       # LLM generation (Ollama/OpenAI)
│   │   └── pipeline.py        # Pipeline orchestrator
│   └── main.py                # App entry point + lifespan
├── data/                      # Local storage (gitignored)
├── usage_guide.md             # Detailed usage instructions
├── implementation_phases.md   # Architectural breakdown
├── requirements.txt
└── .env.example
```

## Quick Start

### 1. Prerequisites
- Python 3.12+
- MongoDB 7.0+ (running locally)
- Ollama (running locally) or OpenAI API Key

### 2. Setup

```bash
# Clone and setup
cd hdfc-rag
python -m venv venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env: Set DATABASE_TYPE=mongodb and LLM_PROVIDER=ollama
```

### 3. Run

```bash
# Start MongoDB (if not running)
brew services start mongodb-community@7.0

# Start server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000](http://localhost:8000)

## Documentation

- [Usage Guide](usage_guide.md) — How to upload, query, and manage documents.
- [Implementation Phases](implementation_phases.md) — Detailed breakdown of the 3-phase build.

## API Usage Example

### Query via cURL
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the annual fee for Infinia card?",
    "top_k": 5
  }'
```

## Configuration

| Variable              | Default               | Description                       |
|-----------------------|-----------------------|-----------------------------------|
| `DATABASE_TYPE`       | `mongodb`             | `mongodb` or `postgres`           |
| `MONGODB_URL`         | `mongodb://localhost:27017`| MongoDB connection URL       |
| `LLM_PROVIDER`        | `ollama`              | `ollama`, `openai`, or `azure`    |
| `AZURE_OPENAI_API_KEY`| —                     | Azure API key                     |
| `AZURE_OPENAI_ENDPOINT`| —                    | Azure Endpoint URL                |
| `LLM_MODEL`           | `qwen2.5:7b`          | Model name for Ollama/OpenAI      |
| `CHROMA_PERSIST_DIR`  | `./data/chroma_db`    | Vector store directory            |
