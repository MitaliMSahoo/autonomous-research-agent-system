import sys
from pathlib import Path

from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from app.llm import get_llm_model
from app.tools import echo_modifier_tool, get_tavily_search_tool



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

    agent = build_react_agent()
    config = {"configurable": {"thread_id": "react-agent-demo"}}

    result = agent.invoke({
    "messages": [
        HumanMessage(content="Please echo the text: Hello ReAct agent!")
    ]
    }, config=config)
    print(f"\nAgent response (last message): [Example 1]")
    print(result["messages"][-1].content)
    print()

        # Example 2: Reasoning then tool call
    print("[Example 2] Agent reasons before calling tool")
    print("-" * 70)
    
    result2 = agent.invoke({
        "messages": [
            HumanMessage(content="What is 2+2? Then echo the answer.")
        ]
    }, config=config)
    
    print(f"\nAgent response (last message):")
    print(result2["messages"][-1].content)
    print()
    
    # Example 3: Question that doesn't need a tool
    print("[Example 3] Question that doesn't require a tool")
    print("-" * 70)
    
    result3 = agent.invoke({
        "messages": [
            HumanMessage(content="What is your name?")
        ]
    }, config=config)
    
    print(f"\nAgent response (last message):")
    print(result3["messages"][-1].content)
    print()
    
