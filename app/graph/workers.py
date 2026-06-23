"""Worker nodes for parallel sub-question research."""
import json
from typing import Dict, Any
from langgraph.types import Send
from app.tools import tavily_search, scrape_url


async def search_worker(state: Dict[str, Any]) -> dict:
    """
    Worker node: takes one sub_question, researches it using Tavily + scraper.
    Runs in parallel — one instance per sub-question.
    """
    sub_question = state["sub_question"]

    # 1. Search web for this sub-question (top 5 results)
    search_results = await tavily_search.ainvoke({
        "query": sub_question,
        "max_results": 5
    })

    # 2. Scrape top 3 results for full content
    scraped = []
    for r in search_results[:3]:
        content = await scrape_url.ainvoke({"url": r["url"]})
        scraped.append({
            "url": r["url"],
            "title": r["title"],
            "content": content[:2000]
        })

    # 3. Return result — operator.add in state merges these automatically
    return {
        "worker_results": [{
            "sub_question": sub_question,
            "summary": json.dumps(scraped, indent=2),
            "sources": [r["url"] for r in search_results],
            "confidence": 0.8,
            "retries": 0
        }]
    }


def route_to_workers(state: Dict[str, Any]) -> list[Send]:
    """
    Orchestrator: conditional edge function.
    Reads approved sub_questions → fans out to parallel search_worker instances.
    """
    sub_questions = state.get("sub_questions", [])
    if not sub_questions:
        return [Send("__end__", {})]  # No work to do

    # One Send() per sub-question → runs in parallel
    return [
        Send("search_worker", {"sub_question": q})
        for q in sub_questions
    ]