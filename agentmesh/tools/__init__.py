# file: agentmesh/tools/__init__.py
"""Tool registry and implementations."""

from agentmesh.tools.file_io import read_file, write_file
from agentmesh.tools.knowledge_base import KnowledgeBase, query_knowledge_base
from agentmesh.tools.registry import ToolDef, ToolRegistry, tool
from agentmesh.tools.run_python import run_python
from agentmesh.tools.search_web import search_web


def create_default_registry() -> ToolRegistry:
    """Create a ToolRegistry with all built-in tools registered."""
    registry = ToolRegistry()
    registry.register(search_web)
    registry.register(run_python)
    registry.register(read_file)
    registry.register(write_file)
    registry.register(query_knowledge_base)
    return registry


__all__ = [
    "ToolDef",
    "ToolRegistry",
    "tool",
    "create_default_registry",
    "search_web",
    "run_python",
    "read_file",
    "write_file",
    "query_knowledge_base",
    "KnowledgeBase",
]