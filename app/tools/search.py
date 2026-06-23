from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
import os
from typing import List, Dict
from langchain_core.tools import tool
from tavily import TavilyClient

_tavily_client = None

def _get_tavily_client() -> TavilyClient:
    global _tavily_client
    if _tavily_client is None:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY not set")
        _tavily_client = TavilyClient(api_key=api_key)
    return _tavily_client

@tool
def tavily_search(query: str, max_results: int = 5) -> List[Dict]:
    """Search the web using Tavily. Returns top {max_results} results with title, url, and snippet."""
    client = _get_tavily_client()
    response = client.search(query=query, max_results=max_results)
    return [
        {
            "title": r.get("title"),
            "url": r.get("url"),
            "snippet": r.get("content"),
        }
        for r in response.get("results", [])
    ]

 
@tool
def echo_modifier_tool(text: str) -> str:
    """Echo the input text."""
    return f"Echo Response: {text}"
 

def get_tavily_search_tool(max_results: int = 3) -> TavilySearchResults:
    """
    Initializes and returns the live web-browsing Tavily client.
    """
    return TavilySearchResults(max_results=max_results)

if __name__ == "__main__":
    result = tavily_search.invoke({"query": "nuclear fusion latest", "max_results": 3})
    for r in result:
        print(f"\n🔗 {r['title']}")
        print(f"   {r['url']}")
        print(f"   {r['snippet'][:200]}...")