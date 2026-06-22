import sys
from pathlib import Path

from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from app.llm import get_llm_model
from app.tools import echo_modifier_tool, get_tavily_search_tool
from app.graph.planner import planner_node
from langgraph.graph import StateGraph, START, END
from app.graph.state import ResearchState

def build_research_graph():
    builder = StateGraph(ResearchState)
    builder.add_node("planner", planner_node)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", END)  # Temporary - will add workers/reporter later
    
    memory_checkpointer = MemorySaver()
    return builder.compile(checkpointer=memory_checkpointer)

def build_react_agent():

    model = get_llm_model()

    tools_list = [echo_modifier_tool, get_tavily_search_tool()]

    memory_checkpointer = MemorySaver()

    agent_graph = create_agent(
        model=model,
        tools=tools_list,
        checkpointer=memory_checkpointer
    )

    return agent_graph


if __name__ == "__main__":
    import asyncio
    
    print("=" * 70)
    print("[Test] Building Research Graph")
    print("=" * 70)
    
    agent = build_research_graph()
    print("✓ Research graph built successfully")
    print(f"✓ Graph nodes: {list(agent.nodes.keys())}")
    print()
    
    # Test with a sample research query
    print("=" * 70)
    print("[Test] Invoking graph with sample query")
    print("=" * 70)
    
    config = {"configurable": {"thread_id": "test-research-001"}}
    sample_state = {
        "query": "What are the latest developments in quantum computing?",
        "sub_questions": [],
        "worker_results": [],
        "results": [],
        "eval_scores": None,
        "job_id": "test-001",
        "status": "started"
    }
    
    async def test_graph():
        try:
            result = await agent.ainvoke(sample_state, config=config)
            print("✓ Graph completed successfully (no interrupts)")
            print(f"✓ Final state keys: {list(result.keys())}")
            print(f"✓ Status: {result.get('status')}")
        except Exception as e:
            print(f"✗ Error during ainvoke: {e}")
            import traceback
            traceback.print_exc()
        
        # Check checkpoint state to see if graph paused at interrupt()
        try:
            checkpoint = agent.get_state(config)
            print()
            print("Checkpoint State:")
            if checkpoint.tasks:
                print("✓ Graph is PAUSED at interrupt() - This is expected!")
                print(f"  Pending tasks: {len(checkpoint.tasks)}")
                
                # Extract the interrupt payload (the generated questions)
                interrupt_task = checkpoint.tasks[0]
                if hasattr(interrupt_task, 'interrupts') and interrupt_task.interrupts:
                    interrupt_payload = interrupt_task.interrupts[0].value
                    print(f"  Interrupt payload (generated questions): {interrupt_payload}")
                else:
                    print(f"  No interrupt payload found")
            else:
                print("✓ Graph has no pending interrupts")
                print(f"  Status: {checkpoint.values.get('status')}")
                print(f"  Sub-questions: {checkpoint.values.get('sub_questions')}")
        except Exception as e:
            print(f"Note: Could not get checkpoint state: {e}")
    
    asyncio.run(test_graph())
    print()
    print("=" * 70)
    print("[Test] Complete")
    print("=" * 70)
