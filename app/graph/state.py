from typing import Annotated, Any, Optional, TypedDict
import operator

class EvalScores(TypedDict):
    groundedness: int
    task_success: int
    hallucination_rate: float
    citation_accuracy: int
    reasoning: str

class workerResult(TypedDict):
    """What each search worker returns"""
    sub_question: str
    summary: str
    sources: list[str]
    confidence: float
    retries: int

class ResearchState(TypedDict):
    query: str
    sub_questions: list
    worker_results: Annotated[list[dict[str, Any]], operator.add]
    results: list
    eval_scores: Optional[EvalScores]
    job_id: str
    status: str


def echo(state: ResearchState):
    return state




