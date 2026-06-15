"""
For planner, we would need to implement a few things:
get the reseerachState, 


will take the query -> decouple the questions (based on what?)(who will decouple? via LLM API) -> return it 

we need constructor:
Planner should have what - understand it.



"""

from typing import List
from pydantic import BaseModel
from app.graph.state import ResearchState
from app.llm.router import get_llm_model

class ResearchPlan(BaseModel):
    sub_questions: List[str]
    status: str = "pending"



async def planner_node(state: ResearchState) -> dict:
    """
    Planner node: breaks the user's query into sub-questions using the LLM.
    The LLM args (model, temperature, provider) come from .env via get_llm_model().
    """


    # 1. Get LLM instance (auto-configured from .env)
    llm = get_llm_model()

    # 2. Bind structured output so the model returns a ResearchPlan
    structured_llm = llm.with_structured_output(ResearchPlan)

    # 3. Invoke with the system prompt + user query
    system_prompt = """You are a research assistant. Your job is to break down the user's research question into 2-5 sub-questions that can be independently researched to answer the main question.
    The sub-questions should collectively cover the key aspects needed to answer the main question, without being too broad or too narrow.
    When generating sub-questions, consider the following:
    - Relevance: Each sub-question should be directly relevant to the main question and contribute to answering it.
    - Clarity: Sub-questions should be clearly phrased and unambiguous.
    - Coherence: Sub-questions should be related to each other and form a cohesive whole.
    - Complexity: The number of sub-questions should be appropriate for the complexity of the main question.
    - Diversity: The sub-questions should cover a wide range of topics and perspectives.
    """
    messages = [
        ("system", system_prompt),
        ("human", state["query"]),
    ]
    plan: ResearchPlan = await structured_llm.ainvoke(messages)

    # 4. Return sub-questions to update the graph state
    return {
        "sub_questions": plan.sub_questions,
        "status": "planned"
    }


if __name__ == "__main__":
    import asyncio

    # Test the planner node with a sample query
    test_state: ResearchState = {
        "query": "What is the impact of AI on healthcare?",
        "sub_questions": [],
        "worker_results": [],
        "results": [],
        "eval_scores": None,
        "job_id": "test-123",
        "status": "started"
    }

    result = asyncio.run(planner_node(test_state))
    print("\n=== Planner Result ===")
    print(f"Status: {result['status']}")
    print(f"Sub-questions ({len(result['sub_questions'])}):")
    for i, q in enumerate(result["sub_questions"], 1):
        print(f"  {i}. {q}")


