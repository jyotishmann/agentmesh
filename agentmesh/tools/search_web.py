# file: agentmesh/tools/search_web.py
"""Web search tool using DuckDuckGo."""

import logging

from duckduckgo_search import DDGS

from agentmesh.config import settings
from agentmesh.tools.registry import tool

logger = logging.getLogger(__name__)


def _format_results(results: list[dict], max_results: int = 5) -> str:
    """Format DuckDuckGo results into a numbered text list."""
    if not results:
        return "No results found."

    lines = []
    for i, r in enumerate(results[:max_results], 1):
        title = r.get("title", "No title")
        body = r.get("body", r.get("snippet", "No description"))
        url = r.get("href", r.get("link", ""))
        lines.append(f"{i}. {title}\n   {body}\n   URL: {url}")

    return "\n\n".join(lines)


@tool(
    name="search_web",
    description="Search the web for current information using DuckDuckGo. "
    "Returns the top search results with titles, snippets, and URLs. "
    "Use this when you need up-to-date information or facts you are unsure about.",
    parameters={
        "query": {
            "type": "str",
            "description": "The search query string (be specific for better results)",
        }
    },
)
def search_web(query: str) -> str:
    """Execute a web search and return formatted results."""
    if not query or not query.strip():
        return "Error: Empty search query."

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        return _format_results(results)

    except Exception as e:
        error_msg = f"Error: Web search failed — {type(e).__name__}: {str(e)}"
        logger.error(error_msg)
        return error_msg