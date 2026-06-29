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
from contextlib import asynccontextmanager
from app.storage.db import init_pool, close_pool
from app.storage.migrations import run_migrations

@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = await init_pool()
    await run_migrations(pool)
    yield
    await close_pool()

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
debug=settings.DEBUG,
lifespan=lifespan
)

agent = build_research_graph()

#
@app.post("/research")
async def start_research(request: ResearchRequest) -> dict:
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "running", 
        "result": None,
        "error": None,
        "query": request.query
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
            result=checkpoint_state.values.get("worker_results", []),
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
async def get_report(job_id: str) -> dict:
    from app.storage.reports import get_report

    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    # 1. Grab worker_results from the graph checkpoint
    config = {"configurable": {"thread_id": job_id}}
    graph_data = {}
    try:
        state = agent.get_state(config)
        graph_data = state.values
    except Exception:
        pass

    worker_results = graph_data.get("worker_results", job.get("result", {}).get("worker_results", []))

    # 2. Grab summaries from the report_store DB
    try:
        db_reports = await get_report(job_id)
    except Exception:
        db_reports = []

    return {
        "job_id": job_id,
        "query": graph_data.get("query") or job.get("result", {}).get("query"),
        "status": job["status"],
        "sub_questions": graph_data.get("sub_questions", []),
        "worker_results": worker_results,
        "db_reports": db_reports,
    }

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
        # Resume the graph from the interrupt point
        # Command(resume=...) passes the approved questions back to interrupt()
        # Run in background so endpoint returns immediately
        async def _resume_and_store():
            try:
                result = await agent.ainvoke(
                    Command(resume=approved_questions),
                    config=config
                )
                jobs[job_id]["status"] = "completed"
                jobs[job_id]["result"] = {
                    "query": jobs[job_id].get("query", ""),
                    "sub_questions": result.get("sub_questions", []),
                    "worker_results": result.get("worker_results", []),
                }
                print(f"[{job_id}] Research complete — {len(result.get('worker_results', []))} worker results")
            except Exception as e:
                import traceback as tb
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["error"] = str(e)
                print(f"[{job_id}] Resume failed: {e}")
                tb.print_exc()

        asyncio.create_task(_resume_and_store())

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
    print("hello ok")
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
        
        # Check if graph was interrupted (awaiting approval)
        try:
            checkpoint = agent.get_state(config)
            if checkpoint.tasks:
                # Paused at interrupt() — waiting for approval
                jobs[job_id]["status"] = "awaiting_approval"
                print(f"[{job_id}] Paused at interrupt — awaiting approval")
                return
        except Exception:
            pass

        # Graph completed successfully (no interrupts)
        print(f"[{job_id}] Research agent completed successfully")
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = {
            "query": query,
            "sub_questions": result.get("sub_questions", []),
            "worker_results": result.get("worker_results", []),
            "status": result.get("status", "unknown")
        }
        print(f"[{job_id}] Research complete — {len(result.get('worker_results', []))} worker results")
        
    except Exception as e:
        # Handle unexpected errors
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

