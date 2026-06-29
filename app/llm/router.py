"""
Routes to LLM Providers
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from app.config import settings
import os

load_dotenv()

def get_llm_model():
    """
    Returns the configured LLM based on LLM_PROVIDER env var.
    
    Defaults to Ollama (local, free).
    Set LLM_PROVIDER=openai or LLM_PROVIDER=anthropic to switch.
    """
    
    if settings.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=settings.LLM_TEMPERATURE
        )
    
    elif settings.LLM_PROVIDER == "groq":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.GROQ_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            base_url="https://api.groq.com/openai/v1",
            api_key=settings.GROQ_API_KEY,
        )
    
    elif settings.LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.ANTHROPIC_MODEL,
            temperature=settings.LLM_TEMPERATURE
        )
    
    elif settings.LLM_PROVIDER == "ollama":
        from langchain_community.llms import Ollama
        return Ollama(
            model=settings.OLLAMA_MODEL,
            temperature=settings.LLM_TEMPERATURE
        )
    
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.LLM_PROVIDER}")

# ---------------------------------------------------------------------------
# Embedding — tries Ollama first, falls back to OpenAI
# ---------------------------------------------------------------------------

async def _ollama_embed(texts: list[str]) -> list[list[float]]:
    import httpx
    timeout = httpx.Timeout(120.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            "http://localhost:11434/api/embed",  # batch endpoint
            json={"model": "nomic-embed-text", "input": texts}
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]


_embed_client = None

def _get_embed_client():
    global _embed_client
    if _embed_client is None:
        from openai import AsyncOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set. Cannot create embedding client.")
        _embed_client = AsyncOpenAI(api_key=api_key)
    return _embed_client

EMBED_DIMS = 768  # nomic-embed-text; change to 1536 if using OpenAI


async def embed_text(text: str) -> list[float]:
    """Single text → embedding vector (uses Ollama locally)."""
    result = await _ollama_embed([text])
    return result[0]


async def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Multiple texts → embedding vectors (uses Ollama locally)."""
    return await _ollama_embed(chunks)