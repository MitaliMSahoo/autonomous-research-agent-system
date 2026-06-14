# app/main.py
from fastapi import FastAPI
from typing import TypedDict
from app.config import settings
from langgraph.graph import StateGraph, START, END

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    debug=settings.DEBUG
)

class EchoState(TypedDict):
    msg: str

@app.get("/")
def read_root():
    return {
        "status": "online",
        "os": "Windows",
        "database_configured": bool(settings.DATABASE_URL)
    }


def echo_node(state: EchoState) -> dict:
    original_msg = state["msg"]
    echoed_msg = "Echo: " + original_msg
    print(f"[echo_node] received: '{original_msg}'")
    print(f"[echo_node] returning: '{echoed_msg}'")
    return {"echoed_msg": echoed_msg}


def shout_node(state: EchoState) -> dict:
    original_msg = state["msg"]
    shouted_msg = original_msg.upper()
    print(f"[shout_node] received: '{original_msg}'")
    print(f"[shout_node] returning: '{shouted_msg}'")
    return {"shouted_msg": shouted_msg}



builder = StateGraph(EchoState)

builder.add_node("echo_node", echo_node)
builder.add_node("shout_node", shout_node)

builder.add_edge(START, "echo_node")
builder.add_edge("echo_node", "shout_node")
builder.add_edge("shout_node", END)

builder.set_entry_point("echo_node")


graph = builder.compile()

print("\n--- invoking graph ---\n")
 
result = graph.invoke({"message": "hello world"})
 
print(f"\n--- final state ---")
print(f"result: {result}")
print(f"message field: {result['message']}")