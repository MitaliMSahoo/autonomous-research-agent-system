# app/main.py
import traceback
import uuid

from fastapi import FastAPI
from typing import Optional, TypedDict

from pydantic import BaseModel
import asyncio
from fastapi import FastAPI, HTTPException
from app.config import settings
from app.graph.react_agent import build_react_agent, build_research_graph
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

jobs = {} # In-memory job store for demonstration

class EchoState(TypedDict):
    msg: str

class ResearchRequest(BaseModel):
    """User's research question."""
    query: str

class JobStatus(BaseModel):
    job_id: str 
    status: str # pending | running | completed | failed | awaiting_approval
    result: Optional[dict] = None
    error:  Optional[str] = None
    query: Optional[str] = None
    messages: Optional[list] = None
    interrupted: Optional[list] = None  # Generated sub-questions waiting for approval



app = FastAPI(
title=settings.PROJECT_NAME,
version=settings.VERSION,
debug=settings.DEBUG
)

agent = build_research_graph()

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
    print(jobs)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    metadata = job
    config = {"configurable": {"thread_id": job_id}}
    try:
        checkpoint_state = agent.get_state(config)
        
        interrupted_payload = None
        if checkpoint_state.tasks:
            # Graph is paused at interrupt()
            status = "awaiting_approval"
            # Extract the interrupt payload
            interrupted_payload = checkpoint_state.tasks[0].interrupts[0].value
        elif checkpoint_state.next:
            # Graph is running
            status = "running"
        else:
            # Graph is done
            status = "done"
        
        messages = checkpoint_state.values.get("messages", [])
        messages_formatted = [
            {
                "type": type(msg).__name__,
                "content": msg.content
            }
            for msg in messages
        ] if messages else []

        return JobStatus(
            job_id=job_id,
            status=status,
            query=checkpoint_state.values.get("query"),
            messages=messages_formatted,
            error=metadata.get("error"),
            interrupted=interrupted_payload  # ← Return interrupt payload if paused
        )
        
    except Exception as e:
        # If checkpoint doesn't exist yet, return metadata status
        return JobStatus(
            job_id=job_id,
            status=metadata["status"],
            error=metadata.get("error")
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
@app.post("/research/{job_id}/approve")
async def approve_plan(job_id: str, body: dict) -> dict:
    """
    POST /research/{job_id}/approve
    
    User approves the plan (or edits it) and resumes the graph.
    This is called when GET /status returns status="awaiting_approval".
    
    Body should contain: {"approved_questions": [...]}
    """
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    config = {"configurable": {"thread_id": job_id}}
    approved_questions = body.get("approved_questions", [])
    
    try:
        # Resume the graph with the approved questions
        # This is one call that resumes from interrupt() point
        asyncio.create_task(
            agent.ainvoke(
                Command(resume=approved_questions),
                config=config
            )
        )
        
        return {
            "status": "approved",
            "job_id": job_id,
            "message": "Graph resumed with approved questions"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
 

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
    """
    Run the research agent in the background.
    Handles initial invocation and waits for user approval at interrupt().
    """
    try: 
        config = {"configurable": {"thread_id": job_id}}

        result = await agent.ainvoke({
            "query": query,
            "sub_questions": [],
            "worker_results": [],
            "results": [],
            "eval_scores": None,
            "job_id": job_id,
            "status": "started"
        }, config=config)
        
        # Graph completed successfully (no interrupts)
        print(f"[{job_id}] Research agent completed successfully")
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = {
            "query": query,
            "sub_questions": result.get("sub_questions", []),
            "status": result.get("status", "unknown")
        }
        print(f"[{job_id}] Research complete")
        
    except Exception as e:
        # Handle unexpected errors (not GraphInterrupt)
        # GraphInterrupt is NOT thrown - graph pauses instead via checkpoint
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        print(f"[{job_id}] Research failed: {e}")
        traceback.print_exc()


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