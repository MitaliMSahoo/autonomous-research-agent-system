"""Worker nodes for parallel sub-question research."""
import json
from typing import Dict, Any
from langgraph.types import Send
from app.tools import tavily_search, scrape_url
from app.utils import chunk_text
from app.llm.router import embed_text, embed_chunks, get_llm_model
from app.storage.pgvector import store_chunks, retrieve_top_k
from app.storage.reports import save_report


async def search_worker(state: Dict[str, Any]) -> dict:
    """
    Worker node: takes one sub_question, researches it using Tavily + scraper.
    Runs in parallel — one instance per sub-question.
    """
    sub_question = state["sub_question"]
    job_id = state["job_id"]

    # 1. Search web for this sub-question (top 5 results)
    search_results = await tavily_search.ainvoke({
        "query": sub_question,
        "max_results": 5
    })

    # 2. Scrape top 5 results — keep only successful scrapes
    scraped = []
    for r in search_results[:5]:
        content = await scrape_url.ainvoke({"url": r["url"]})

        # Skip scraper errors — treat as failed scrape
        if content.startswith("Error scraping") or content == "Could not extract content from this page.":
            # Use Tavily snippet as fallback instead
            snippet = r.get("snippet", "")
            if snippet and len(snippet) > 50:
                scraped.append({
                    "url": r["url"],
                    "title": r["title"],
                    "content": snippet[:2000]
                })
            continue

        scraped.append({
            "url": r["url"],
            "title": r["title"],
            "content": content[:2000]
        })

        # Stop once we have 3 good scrapes
        if len([s for s in scraped if not s["content"].startswith("Error")]) >= 3:
            break

    # 3. Calc confidence
    confidence = 0.0
    if search_results:
        confidence += min(len(search_results) / 5, 1.0) * 0.3
        successful_scrapes = [s for s in scraped if len(s["content"]) > 200 and not s["content"].startswith("Error")]
        if successful_scrapes:
            confidence += min(len(successful_scrapes) / 3, 1.0) * 0.7
    confidence = round(min(confidence, 1.0), 2)

    # 4. Summarize — skip bad content entirely
    good_content = [s["content"] for s in scraped if s["content"] and not s["content"].startswith("Error")]
    if not good_content:
        # No scraped content at all — use Tavily snippets
        good_content = [r.get("snippet", "") for r in search_results if r.get("snippet")]
    full_text = "\n\n".join(good_content)
    summary = await _summarize(job_id, sub_question, full_text)

    # 5. Save to report_store
    await save_report(job_id, sub_question, summary)

    # 6. Determine retry count
    retry_count = state.get("_retries", 0)
    original_q = state.get("_original", sub_question)

    # 7. Return result — raw text never leaves worker, only summary
    return {
        "worker_results": [{
            "sub_question": original_q,
            "summary": summary,                         # ← compressed, not raw
            "sources": [r["url"] for r in search_results],
            "confidence": confidence,
            "retries": retry_count
        }]
    }


async def _summarize(job_id: str, sub_question: str, scraped_text: str) -> str:
    """chunk → embed → store → retrieve top-5 → LLM compress to 2-3 sentences."""
    if not scraped_text.strip():
        return "No content found for this sub-question."

    # Chunk
    chunks = chunk_text(scraped_text)

    # Embed all chunks in one API call
    embeddings = await embed_chunks(chunks)

    # Store in pgvector
    await store_chunks(job_id, sub_question, chunks, embeddings)

    # Embed sub_question as query vector
    query_embedding = await embed_text(sub_question)

    # Retrieve top-5 most relevant chunks
    relevant_chunks = await retrieve_top_k(job_id, query_embedding, top_k=5)

    # LLM compress
    context = "\n\n".join(relevant_chunks)
    llm = get_llm_model()
    prompt = (
        f"Based on the following research excerpts, answer this question in "
        f"2-3 sentences only. Be specific and factual.\n\n"
        f"Question: {sub_question}\n\n"
        f"Excerpts:\n{context}"
    )
    response = await llm.ainvoke([("human", prompt)])
    return response.content


def route_retry(state):
    retries = state.get("retries", [])
    if retries:
        return [Send("search_worker", {
            "sub_question": r["reformulated_query"],
            "_original": r["original_sub_question"],
            "_retries": r["retries"]
        }) for r in retries]
    return "__end__"


def route_to_workers(state: Dict[str, Any]) -> list[Send]:
    sub_questions = state.get("sub_questions", [])
    if not sub_questions:
        return "__end__"
    return [
        Send("search_worker", {
            "sub_question": q,
            "job_id": state["job_id"]   # ← pass job_id into each worker
        })
        for q in sub_questions
    ]