import logging
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows consoles
if sys.platform.startswith("win"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def configure_clean_logging() -> None:
    """
    Suppresses noisy third-party INFO logs (httpx, google_genai AFC notices,
    LlamaIndex index loader chatter, and OpenInference event handler tracebacks)
    so CLI and Streamlit outputs remain clean.
    """
    noisy_loggers = (
        "httpx",
        "httpcore",
        "urllib3",
        "google_genai",
        "google_genai.models",
        "google.genai",
        "llama_index.core.indices.loading",
        "llama_index.core.indices.struct_store.sql_query",
        "llama_index.core.indices.struct_store.sql_retriever",
        "openinference.instrumentation.llama_index",
        "openinference.instrumentation.llama_index._handler",
    )
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)

    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"openinference.*")
    warnings.filterwarnings("ignore", category=UserWarning, module=r"pydantic.*")


configure_clean_logging()

# Load .env file from project root
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _env_bool(key: str, default: bool = True) -> bool:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("true", "1", "yes", "on")


_STORAGE_REL = os.getenv("STORAGE_DIR", "storage")
_DOCS_REL = os.getenv("DOCS_DIR", "data/unstructured")


@dataclass
class AppConfig:
    """Central configuration for Nexus Specialized Document Agent."""

    # Project Directories
    root_dir: Path = ROOT_DIR
    data_dir: Path = ROOT_DIR / "data"
    unstructured_dir: Path = ROOT_DIR / _DOCS_REL
    structured_dir: Path = ROOT_DIR / "data" / "structured"
    db_path: Path = (
        ROOT_DIR
        / os.getenv("SQLITE_DB_PATH", "data/structured/ecommerce_retail.db")
    )
    storage_dir: Path = ROOT_DIR / _STORAGE_REL
    chat_history_dir: Path = ROOT_DIR / _STORAGE_REL / "chat_history"

    # Model Settings (Multi-Model: Google Gemini, DeepSeek, Ollama)
    llm_provider: str = os.getenv("LLM_PROVIDER", "google").strip().lower()
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "").strip()
    llm_model: str = os.getenv("LLM_MODEL", "gemini-2.5-flash-lite").strip()
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2-preview").strip()
    temperature: float = _env_float("TEMPERATURE", 0.1)
    agent_mode: str = os.getenv("AGENT_MODE", "function_calling").strip()

    # DeepSeek Configuration
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "").strip()
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").strip()
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()

    # Ollama Configuration
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:latest").strip()
    ollama_num_gpu: int = _env_int("OLLAMA_NUM_GPU", 0)
    ollama_context_window: int = _env_int("OLLAMA_CONTEXT_WINDOW", 4096)
    enable_ollama_fallback: bool = _env_bool("ENABLE_OLLAMA_FALLBACK", True)

    # Arize Phoenix Observability & Tracing Configuration
    enable_phoenix_tracing: bool = _env_bool("ENABLE_PHOENIX_TRACING", True)
    phoenix_base_url: str = os.getenv("PHOENIX_BASE_URL", "http://localhost:6006").strip()
    phoenix_collector_endpoint: str = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces").strip()
    phoenix_project_name: str = os.getenv("PHOENIX_PROJECT_NAME", "nexus-document-agent").strip()

    def ensure_directories(self) -> None:
        """Create necessary directories if they do not exist."""
        self.unstructured_dir.mkdir(parents=True, exist_ok=True)
        self.structured_dir.mkdir(parents=True, exist_ok=True)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.chat_history_dir.mkdir(parents=True, exist_ok=True)

    def is_api_key_configured(self, provider: str = "google", key: str = "") -> bool:
        """Check if a valid API key is configured for the given provider."""
        p = provider.lower()
        if p == "ollama":
            return True  # Ollama is local, doesn't require an external API key
        if p == "deepseek":
            effective = (key or self.deepseek_api_key).strip()
            return bool(effective and effective != "your_deepseek_api_key_here")
        # Default to google
        effective = (key or self.gemini_api_key).strip()
        return bool(effective and effective != "your_gemini_api_key_here")


config = AppConfig()
config.ensure_directories()
