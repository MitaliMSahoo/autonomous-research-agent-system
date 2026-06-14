from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults


@tool
def echo_modifier_tool(text: str) -> str:
    """Echo the input text."""
    return f"Echo Response: {text}"


def get_tavily_search_tool(max_results: int = 3) -> TavilySearchResults:
    """Create a Tavily search tool instance."""
    return TavilySearchResults(max_results=max_results)
