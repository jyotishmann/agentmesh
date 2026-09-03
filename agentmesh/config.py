# file: agentmesh/config.py
"""Centralised configuration. All tunables live here."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings, read from environment variables and .env file."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # ── Model Configuration ──────────────────────────────────────────
    main_model_name: str = Field(
        default="Qwen/Qwen2.5-3B-Instruct",
        description="HuggingFace model ID for Planner and Critic agents",
    )
    specialist_model_name: str = Field(
        default="Qwen/Qwen2.5-1.5B-Instruct",
        description="HuggingFace model ID for Specialist agents",
    )
    embedding_model_name: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="HuggingFace model ID for embeddings",
    )

    # ── Generation Parameters ───────────────────────────────────
    temperature: float = Field(default=0.7, description="Sampling temperature")
    max_new_tokens: int = Field(default=1024, description="Max tokens to generate")
    top_p: float = Field(default=0.9, description="Nucleus sampling threshold")

    # ── Agent Parameters ────────────────────────────────────────
    max_tool_calls_per_agent: int = Field(
        default=5, description="Max tool calls per specialist per sub-task"
    )
    max_total_tool_calls: int = Field(
        default=20, description="Hard cap on total tool calls per task"
    )
    max_revision_cycles: int = Field(
        default=2, description="Max critic-specialist revision loops"
    )
    conversation_buffer_size: int = Field(
        default=20, description="Max messages in conversation buffer"
    )

    # ── Tool Configuration ──────────────────────────────────────
    sandbox_dir: str = Field(
        default="data/sandbox", description="Directory for file I/O tool"
    )
    knowledge_base_dir: str = Field(
        default="data/knowledge_base", description="Directory for KB documents"
    )
    python_timeout: int = Field(
        default=30, description="Timeout in seconds for Python execution"
    )

    # ── Storage Paths ───────────────────────────────────────────
    db_path: str = Field(
        default="data/agentmesh.db", description="SQLite database path"
    )
    faiss_index_path: str = Field(
        default="data/memory.faiss", description="FAISS index path"
    )
    kb_index_path: str = Field(
        default="data/kb.faiss", description="Knowledge base FAISS index path"
    )

    # ── Server ──────────────────────────────────────────────────
    server_host: str = Field(default="0.0.0.0")
    server_port: int = Field(default=8000)
    device: str = Field(
        default="auto", description="Device: 'auto', 'cuda', or 'cpu'"
    )


# Singleton instance — every module imports this
settings = Settings()
