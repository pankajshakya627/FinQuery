# Component Guide: FinQuery Technical Reference

This document provides a comprehensive mapping of every folder and significant file in the **FinQuery AI** repository.

## 📂 Repository Structure

```text
/Users/pankajshakya/Downloads/hdfc-rag
├── app/                  # Main Application Source
│   ├── api/             # FastAPI Route Handlers
│   ├── core/            # Config, Security, Settings
│   ├── db/              # SQL & NoSQL Database Logic
│   ├── models/          # Pydantic & SQLAlchemy Models
│   ├── rag/             # RAG Intelligence (Parser, Chunker, Generator)
│   ├── static/          # CSS, JS, Assets
│   └── templates/       # Jinja2 HTML Templates
├── data/                 # Local Storage (ChromaDB, Uploads)
├── docs/                 # Detailed Documentation & Diagrams
├── scripts/              # Seed & Utility Scripts
├── tests/                # Unit & Integration Tests
├── .env                  # Environment Secret Config
├── Dockerfile            # Container Definition
└── requirements.txt      # Dependency Specification
```

---

## 🏗️ Core Application Components (`app/`)

### 1. RAG Intelligence (`app/rag/`)
*   `pipeline.py`: The central orchestrator. It manages the handoffs between the parser, vector store, and generator.
*   `parser.py`: Uses **IBM Docling** for high-fidelity conversion of PDFs to Markdown.
*   `chunker.py`: Logic for breaking documents into semantic and structural segments.
*   `vector_store.py`: Wrapper for **ChromaDB**, handling vector insertion and cosine similarity searches.
*   `generator.py`: LLM reasoning layer. It manages prompt construction and provider initialization (OpenAI/Ollama).

### 2. API Layer (`app/api/`)
*   `routes.py`: Contains all business endpoints (e.g., `/api/query`, `/api/documents/upload`).
*   `__init__.py`: Exports the central `APIRouter` for the main app.

### 3. Database Layer (`app/db/`)
*   `mongodb.py`: Async connection handler for MongoDB (Query logs, document history).
*   `session.py`: Database session factory for PostgreSQL/SQLAlchemy.
*   `models.py` & `mongo_models.py`: Structural definitions for stored data.

### 4. Configuration (`app/core/`)
*   `config.py`: Centralized **Pydantic Settings** for environment variable validation and global configuration.

---

## 🎨 Frontend Stack (`app/static/` & `app/templates/`)

### 1. HTML Templates (`app/templates/`)
*   `index.html`: The main AI Chat interface, featuring the glassmorphic side-bar and responsive message area.
*   `upload.html`: Admin interface for document ingestion and indexing control.
*   `dashboard.html`: Analytics view for system health and RAG telemetry.

### 2. Assets (`app/static/`)
*   `css/styles.css`: Modern Onyx design system with dual-theme variables and micro-animations.
*   `js/app.js`: Client-side logic for the chat interface, theme switching, and pipeline visualizations.

---

## 🛠️ Scripts & Utilities (`scripts/`)
*   `seed_mitc.py`: A bootstrap script to pre-index critical document sets into the vector database.
*   `clean_db.py` (Optional): Utility for resetting local vector collections.

---

## ⚙️ Deployment & Environment
*   `Dockerfile`: Multi-stage build for production-ready containerization.
*   `requirements.txt`: Locked dependencies to prevent architectural conflicts.
*   `.env.example`: Template for required API keys and provider configurations.

> [!TIP]
> Each component is designed to be **modular**. You can replace the LLM provider in `generator.py` or the vector store in `vector_store.py` without touching the rest of the application.
