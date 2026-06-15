# app/main.py
from fastapi import FastAPI
from typing import TypedDict
from app.config import settings
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


class EchoState(TypedDict):
    msg: str



def echo_node(state: EchoState) -> dict:
    original_msg = state["msg"]
    echoed_msg = "Echo: " + original_msg
    print(f"[echo_node] received: '{original_msg}'")
    print(f"[echo_node] returning: '{echoed_msg}'")
    return {"msg": echoed_msg}


def shout_node(state: EchoState) -> dict:
    original_msg = state["msg"]
    shouted_msg = original_msg.upper()
    print(f"[shout_node] received: '{original_msg}'")
    print(f"[shout_node] returning: '{shouted_msg}'")
    return {"msg": shouted_msg}



builder = StateGraph(EchoState)

builder.add_node("echo_node", echo_node)
builder.add_node("shout_node", shout_node)

builder.add_edge(START, "echo_node")
builder.add_edge("echo_node", "shout_node")
builder.add_edge("shout_node", END)

builder.set_entry_point("echo_node")


checkpoint = MemorySaver()
graph = builder.compile(checkpointer=checkpoint)

print("\n--- invoking graph ---\n")
 
config = {"configurable": {"thread_id": "session-1"}}
print("BEFORE run:", graph.get_state(config))
result = graph.invoke({"msg": "hello world"}, config)
 
print(f"\n--- final state ---")
print(f"result: {result}")
print(f"message field: {result['msg']}")


# Same thread_id — continues the same conversation
result2 = graph.invoke({"msg": "world"}, config)
print(graph.get_state(config))
print(f"result2: {result2}")

# Different thread_id — fresh conversation
config2 = {"configurable": {"thread_id": "session-2"}}
result3 = graph.invoke({"msg": "world"}, config2)
print(graph.get_state(config2))
print(f"result3: {result3}")