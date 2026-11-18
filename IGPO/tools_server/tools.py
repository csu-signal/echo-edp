"""
IGPO Tool Server - Tool Definitions (Web Search Only)
"""

from typing import Dict, Any

# Web Search Tool Definition
WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": "Search the web for information using Google or Bing.",
    "inputs": {
        "query": {
            "type": "array",
            "items": {"type": "string"},
            "description": "A list of search queries (1-3 queries recommended)"
        }
    },
    "example": {"query": ["What is the capital of France?", "Paris population 2024"]}
}


def get_tools(config: Dict[str, Any] = None) -> Dict[str, Dict]:
    """Get available tools (web_search only)."""
    return {"web_search": WEB_SEARCH_TOOL}


def get_tool_names(config: Dict[str, Any] = None) -> list:
    """Get list of available tool names."""
    return ["web_search"]
