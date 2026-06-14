"""
Tools for the Modular Agent
This module defines various tools that can be used by the agent to perform specific tasks."""

from app.tools.search import echo_modifier_tool, get_tavily_search_tool

__all__ = [
    "echo_modifier_tool",
    "get_tavily_search_tool"
]
