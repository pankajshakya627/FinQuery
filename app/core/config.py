"""
Configuration management using Pydantic Settings.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import computed_field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore")

    app_name: str = "HDFC RAG Assistant"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = True

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "hdfc_rag"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres123"
    database_url: str = "postgresql+asyncpg://postgres:postgres123@localhost:5432/hdfc_rag"
    database_url_sync: str = "postgresql://postgres:postgres123@localhost:5432/hdfc_rag"

    # MongoDB settings
    mongodb_url: str = "mongodb://localhost:27017/hdfc_rag"
    database_type: str = "mongodb"  # "postgres" or "mongodb"

    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection_name: str = "hdfc_mitc_docs"

    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    openai_api_key: str = "your-openai-api-key-here"
    llm_provider: str = "openai"  # "openai", "ollama", or "azure"
    llm_model: str = "gpt-4o-mini"  # For ollama: "llama3.2", "mistral", etc.
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    ollama_base_url: str = "http://localhost:11434"

    # Azure OpenAI settings (required if llm_provider="azure")
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_ad_deployment_name: str = ""
    azure_openai_api_version: str = "2023-05-15"

    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k_results: int = 5
    similarity_threshold: float = 0.3

    upload_dir: str = "./data/uploads"
    max_file_size_mb: int = 50

    @computed_field
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


_settings_instance: "Settings | None" = None


def get_settings() -> "Settings":
    """Return the global settings instance (created fresh each process start)."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance
