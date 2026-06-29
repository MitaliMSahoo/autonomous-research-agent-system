"""Evaluator node: checks worker result confidence and retries low-confidence results with reformulated queries."""
import json
from typing import Dict, Any
from app.llm.router import get_llm_model


async def evaluate_workers(state: Dict[str, Any]) -> dict:
    """
    Runs after all search_worker instances finish.
    Checks each worker_result's confidence.
    If confidence < 0.5 and retries < 2, asks LLM to reformulate the query for retry.
    Returns retry payloads that get merged via operator.add into state['retries'].

    IMPORTANT: Only evaluates the LATEST result per sub_question to avoid
    re-triggering retries on old low-confidence attempts.
    """
    worker_results = state.get("worker_results", [])
    retries = []

    # Group results by sub_question, keep only the LAST one per question
    latest_per_question: dict[str, dict] = {}
    for r in worker_results:
        latest_per_question[r["sub_question"]] = r

    # Check only the latest attempt for each sub-question
    low_conf = [
        r for r in latest_per_question.values()
        if r.get("confidence", 0) < 0.5 and r.get("retries", 0) < 2
    ]

    if not low_conf:
        return {"retries": []}  # Nothing to retry

    llm = get_llm_model()

    for r in low_conf:
        original_sub_question = r["sub_question"]
        current_retries = r.get("retries", 0)

        # Ask LLM to reformulate the query for better search results
        prompt = (
            f"The following research sub-question returned poor quality results "
            f"(confidence={r['confidence']}). Please reformulate it to get better, "
            f"more specific search results. Keep the same intent but make it more "
            f"targeted.\n\n"
            f"Original: {original_sub_question}\n\n"
            f"Return ONLY a JSON object: {{\"reformulated\": \"your improved query\"}}"
        )

        resp = await llm.ainvoke([("human", prompt)])
        try:
            data = json.loads(resp.content)
            reformulated = data["reformulated"]
        except (json.JSONDecodeError, KeyError):
            # Fallback: just use original with a prefix
            reformulated = f"detailed information about {original_sub_question}"

        retries.append({
            "reformulated_query": reformulated,
            "original_sub_question": original_sub_question,
            "retries": current_retries + 1
        })

    return {"retries": retries}