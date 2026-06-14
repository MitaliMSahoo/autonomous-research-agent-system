"""
Routes to LLM Providers
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from app.config import settings

load_dotenv()

def get_llm_model(model_name: str = "gpt-4o-mini", temperature: float = 0):
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