"""
For planner, we would need to implement a few things:
get the reseerachState, 


will take the query -> decouple the questions (based on what?)(who will decouple? via LLM API) -> return it 

we need constructor:
Planner should have what - understand it.



"""

import json
from app.graph.state import ResearchState
from app.llm.router import get_llm_model
from langgraph.types import interrupt



async def planner_node(state: ResearchState) -> dict:
    """
    Planner node: breaks the user's query into sub-questions using the LLM.
    The LLM args (model, temperature, provider) come from .env via get_llm_model().
    """


    # 1. Get LLM instance (auto-configured from .env)
    llm = get_llm_model()

    # 2. System prompt for planning
    system_prompt = """You are a research assistant. Your job is to break down the user's research question into 2-5 sub-questions that can be independently researched to answer the main question.
    The sub-questions should collectively cover the key aspects needed to answer the main question, without being too broad or too narrow.
    When generating sub-questions, consider the following:
    - Relevance: Each sub-question should be directly relevant to the main question and contribute to answering it.
    - Clarity: Sub-questions should be clearly phrased and unambiguous.
    - Coherence: Sub-questions should be related to each other and form a cohesive whole.
    - Complexity: The number of sub-questions should be appropriate for the complexity of the main question.
    - Diversity: The sub-questions should cover a wide range of topics and perspectives.

    Return ONLY a valid JSON object with this exact format, no other text:
    {"sub_questions": ["sub-question 1", "sub-question 2", ...]}
    """
    messages = [
        ("system", system_prompt),
        ("human", state["query"]),
    ]
    
    # 3. Invoke LLM and parse JSON response (first run only)
    plan = await llm.ainvoke(messages)
    print(f"\n[Planner Node] Generated sub-questions:", plan)
    plan_data = json.loads(plan.content)
    
    generated_questions = plan_data["sub_questions"]
    print(f"[Planner Node] Pausing for user approval of {len(generated_questions)} sub-questions")
    
    # 4. Pause and wait for user approval
    # On resume: interrupt() returns the approved_questions from Command(resume=...)
    # Code after this line only executes on resume
    approved_questions = interrupt(generated_questions)
    print("moving forward after approval of sub-questions")
    # 5. Return approved sub-questions (executes on resume only)
    return {
        "sub_questions": approved_questions if approved_questions else generated_questions,
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


