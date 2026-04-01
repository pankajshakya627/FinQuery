"""
HDFC RAG Assistant — FastAPI Application Entry Point.

Architecture: FastAPI Backend on Cloud Run
Endpoints: /ingest, /query, /health + Web UI

Run: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from app.core.config import get_settings
from app.db.session import init_database
from app.db.mongodb import MongoDB
from app.rag.pipeline import rag_pipeline
from app.api.routes import (
    health_router,
    document_router,
    query_router,
    stats_router,
)

import structlog

logger = structlog.get_logger(__name__)
settings = get_settings()

# ── App start time for uptime tracking ──
APP_START_TIME = time.time()


# ═══════════════════════════════════════════════════════════════
#  LIFESPAN (startup / shutdown)
# ═══════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown hooks."""
    # ── STARTUP ──
    logger.info("starting_application", app=settings.app_name, env=settings.app_env)

    # 1. Initialize Database based on type
    if settings.database_type == "postgres":
        await init_database()
    
    # 2. Always initialize MongoDB driver
    await MongoDB.connect()

    # 3. Initialize RAG pipeline (embeddings + ChromaDB + LLM)
    rag_pipeline.initialize()

    logger.info("application_ready",
                 host=settings.app_host,
                 port=settings.app_port,
                 db_type=settings.database_type)

    yield

    # ── SHUTDOWN ──
    await MongoDB.disconnect()
    logger.info("shutting_down_application")


# ═══════════════════════════════════════════════════════════════
#  APP INSTANCE
# ═══════════════════════════════════════════════════════════════

app = FastAPI(
    title="HDFC RAG Assistant",
    description=(
        "Production RAG system for HDFC Bank Credit Card MITC document Q&A. "
        "Built with FastAPI + LangChain + ChromaDB + (PostgreSQL/MongoDB)."
    ),
    version="1.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static Files & Templates ──
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


# ── Request Timing Middleware ──
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    process_time = time.time() - start
    response.headers["X-Process-Time"] = f"{process_time:.4f}"
    return response


# ═══════════════════════════════════════════════════════════════
#  REGISTER ROUTERS
# ═══════════════════════════════════════════════════════════════

app.include_router(health_router)
app.include_router(document_router)
app.include_router(query_router)
app.include_router(stats_router)


# ═══════════════════════════════════════════════════════════════
#  WEB UI ROUTES
# ═══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main RAG chat interface."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "app_name": settings.app_name,
    })


@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Serve the document upload page."""
    return templates.TemplateResponse("upload.html", {
        "request": request,
        "app_name": settings.app_name,
    })


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    """Serve the analytics dashboard."""
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "app_name": settings.app_name,
    })
