"""
Minimal RAG Configuration
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve the .env file path relative to this config file
_env_path = Path(__file__).parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_env_path),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection_name: str = "minimal_rag_docs"
    
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    llm_provider: str = "ollama"  # openai | ollama | azure
    llm_model: str = "qwen3.5:9b"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048
    ollama_base_url: str = "http://localhost:11434"

    # Azure OpenAI settings
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_ad_deployment_name: str = ""
    azure_openai_api_version: str = "2023-05-15"

    chunk_size: int = 1500
    chunk_overlap: int = 200
    top_k_results: int = 8
    similarity_threshold: float = 0.15


_settings_instance = None

def get_settings() -> Settings:
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance
