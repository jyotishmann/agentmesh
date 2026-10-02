# file: agentmesh/config.py
"""Centralised configuration. All tunables live here."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings, read from environment variables and .env file."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # ── Model Configuration ─────────────────────────────────────
    main_model_name: str = Field(default="Qwen/Qwen2.5-3B-Instruct")
    specialist_model_name: str = Field(default="Qwen/Qwen2.5-1.5B-Instruct")
    embedding_model_name: str = Field(default="BAAI/bge-small-en-v1.5")

    # ── Generation Parameters ───────────────────────────────────
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_new_tokens: int = Field(default=1024, ge=1, le=4096)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)

    # ── Agent Parameters ────────────────────────────────────────
    max_tool_calls_per_agent: int = Field(default=5, ge=1, le=20)
    max_total_tool_calls: int = Field(default=20, ge=1, le=50)
    max_revision_cycles: int = Field(default=2, ge=0, le=5)
    conversation_buffer_size: int = Field(default=20, ge=1, le=100)

    # ── Tool Configuration ──────────────────────────────────────
    workspace_root: Path = Field(default=Path("data"))
    sandbox_dir: Path = Field(default=Path("data/sandbox"))
    python_timeout: int = Field(default=30, ge=1, le=120)
    file_read_max_chars: int = Field(default=10000, ge=100, le=100000)

    # ── Knowledge Base ──────────────────────────────────────────
    knowledge_base_dir: Path = Field(default=Path("data/knowledge_base"))
    knowledge_base_top_k: int = Field(default=5, ge=1, le=20)

    # ── Storage Paths ───────────────────────────────────────────
    db_path: Path = Field(default=Path("data/agentmesh.db"))
    faiss_index_path: Path = Field(default=Path("data/kb.faiss"))
    kb_index_path: Path = Field(default=Path("data/kb.faiss"))

    # ── Server ──────────────────────────────────────────────────
    server_host: str = Field(default="0.0.0.0")
    server_port: int = Field(default=8000, ge=1, le=65535)
    device: str = Field(default="auto")


# Singleton instance — every module imports this
settings = Settings()
