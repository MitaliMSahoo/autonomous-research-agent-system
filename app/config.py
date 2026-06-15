# app/config.py
import os
import os
from pathlib import Path
from dotenv import load_dotenv
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"

if ENV_FILE_PATH.exists():
    load_dotenv(ENV_FILE_PATH)
else:
    load_dotenv()  # still load from environment/system vars

print(ENV_FILE_PATH, "exists:", ENV_FILE_PATH.exists(), os.getenv("PROJECT_NAME"))
class Settings(BaseSettings):
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "FastAPI default")
    print(PROJECT_NAME)
    VERSION: str = "0.1.0"
    DEBUG: bool = False 
    API_V1_STR: str = "/api/v1"
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "postgresql://user:password@localhost/research_db"
    )
    ENV_TEST_VAR: str = "test" # No default fallback here! Force it to read from .env

    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH, 
        env_file_encoding="utf-8",
        extra="ignore"
    )

    LLM_PROVIDER: Literal["openai", "anthropic", "ollama", "groq"] = os.getenv(
    "LLM_PROVIDER", "ollama"  # default to local Ollama 
    )
    
    # Model names per provider
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama2")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    
    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
    
    # LangSmith tracing
    LANGSMITH_TRACING_V2: bool = os.getenv("LANGSMITH_TRACING_V2", "false").lower() == "true"

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # LLM settings
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0"))  # 0 = deterministic
    
    # Eval thresholds
    EVAL_HALLUCINATION_THRESHOLD: float = 0.2  # fail if > 20% hallucination
    EVAL_TASK_SUCCESS_THRESHOLD: int = 3       # fail if task_success < 3

settings = Settings()
print(f"\n🚀 ENV CHECK: {settings.PROJECT_NAME}\n")