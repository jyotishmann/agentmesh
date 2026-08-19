# file: agentmesh/config.py
"""Centralised configuration. All tunables live here."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings, read from environment variables and .env file."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # ── Model Configuration ──────────────────────────────────────────
    model_name: str = Field(
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