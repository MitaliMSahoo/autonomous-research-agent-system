from langchain.agents import create_react_agent
import tools
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain import hub
from app.llm import get_llm_model
import asyncio
import os
from app.tools import get_tavily_search_tool
from langgraph.checkpoint.memory import MemorySaver



model = get_llm_model()

tools_list = [get_tavily_search_tool()]

memory_checkpointer = MemorySaver()

agent_graph = create_react_agent(
    model=model,
    tools=tools_list,
    checkpointer=memory_checkpointer
)


if __name__ == "__main__":
    print("🤖 ReAct Worker Agent initialized and listening...")
    session_config = {"configurable": {"thread_id": "modular_test_session"}}
    
    while True:
        user_input = input("\nUser 👤: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        events = agent_graph.stream(
            {"messages": [("user", user_input)]}, 
            config=session_config,
            stream_mode="values"
        )
        
        for event in events:
            latest_message = event["messages"][-1]
            if latest_message.type == "ai" and latest_message.content:
                print(f"Agent 🤖: {latest_message.content}")