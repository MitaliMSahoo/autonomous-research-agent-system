# app/main.py
import uuid

from fastapi import FastAPI
from typing import Optional, TypedDict

from pydantic import BaseModel
import asyncio
from fastapi import FastAPI, HTTPException
from app.config import settings
from app.graph.react_agent import build_react_agent
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

jobs = {} # In-memory job store for demonstration

class EchoState(TypedDict):
    msg: str

class ResearchRequest(BaseModel):
    """User's research question."""
    query: str

class JobStatus(BaseModel):
    job_id: str 
    status: str # pending | running | completed | failed
    result: Optional[dict] = None
    error:  Optional[str] = None



app = FastAPI(
title=settings.PROJECT_NAME,
version=settings.VERSION,
debug=settings.DEBUG
)
#Endpoints
# @app.get("/")
# def read_root():
#     return {
#         "status": "online",
#         "os": "Windows",
#         "database_configured": bool(settings.DATABASE_URL)
#     }

#
@app.post("/research")
async def start_research(request: ResearchRequest) -> dict:
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "running", 
        "result": None,
        "error": None
    }

    asyncio.create_task(_run_agent_background(job_id, request.query))
    return {
        "job_id": job_id,
        "status": "running",
        "message": f"Research started. Check status with: GET /status/{job_id}"
    }

#
@app.get("/status/{job_id}")
async def get_status(job_id: str) -> JobStatus:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        result=job["result"],
        error=job["error"]
    )

#
@app.get("/report/{job_id}")
def echo(job_id: str) -> dict:
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] == "pending":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    elif job["status"] == "failed":
        raise HTTPException(status_code=500, detail=f"Job failed with error: {job['error']}")
    elif job["status"] == "running":
        raise HTTPException(status_code=202, detail="Job is still running. Please check back later.")
    else:
        return job["result"]

#
@app.get("/health")
async def health_check() -> dict:
    """
    GET /health
    
    Simple health check endpoint.
    Returns 200 if server is running.
    """
    return {"status": "healthy", "service": "autonomous-research-agent-system"}


async def _run_agent_background(job_id: str, query: str):
    try: 
        agent = build_react_agent()
        config = {"configurable": {"thread_id": job_id}}

        result = agent.invoke({
        "messages": [
            HumanMessage(content=query)
        ]
        }, config=config)
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = {
                "query": query,
                "messages": [
                    {
                        "type": type(msg).__name__,
                        "content": msg.content
                    }
                    for msg in result.get("messages", [])
                ]
            }

        print(f"[{job_id}] Research complete")
    except Exception as e:
        # Mark job as failed
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        print(f"[{job_id}] Research failed: {e}")

    


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


if __name__ == "__main__":

# Build and invoke state graph
    # builder = StateGraph(EchoState)

    # builder.add_node("echo_node", echo_node)
    # builder.add_node("shout_node", shout_node)

    # builder.add_edge(START, "echo_node")
    # builder.add_edge("echo_node", "shout_node")
    # builder.add_edge("shout_node", END)

    # builder.set_entry_point("echo_node")


    # graph = builder.compile()

    # print("\n--- invoking graph ---\n")
    
    # result = graph.invoke({"message": "hello world"})
    
    # print(f"\n--- final state ---")
    # print(f"result: {result}")
    # print(f"message field: {result['message']}")



    import uvicorn
    print(f"Starting server on {settings.HOST}:{settings.PORT}")
    # print(f"Docs available at: http://{settings.HOST}:{settings.PORT}/docs")
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True  # auto-reload on code changes
    )


 
# ── HOW TO TEST THESE ENDPOINTS ──────────────────────────────────
#
# 1. Start the server:
#    python -m app.main
#
# 2. Submit a research query:
#    curl -X POST http://localhost:8000/research \
#      -H "Content-Type: application/json" \
#      -d '{"query": "What is AI?"}'
#
#    Response:
#    {
#      "job_id": "a3f7c-1234-...",
#      "status": "running"
#    }
#
# 3. Check the status:
#    curl http://localhost:8000/status/a3f7c-1234-...
#
#    While running:
#    {
#      "job_id": "a3f7c-1234-...",
#      "status": "running",
#      "result": null
#    }
#
#    When done:
#    {
#      "job_id": "a3f7c-1234-...",
#      "status": "done",
#      "result": {...}
#    }
#
# 4. Get the report:
#    curl http://localhost:8000/report/a3f7c-1234-...
#
# ── INTERACTIVE API DOCS ────────────────────────────────────────
# FastAPI auto-generates interactive docs at:
# http://localhost:8000/docs
#
# Use this to test all endpoints in your browser!