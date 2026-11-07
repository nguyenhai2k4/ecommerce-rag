import os
import sys
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows consoles
if sys.platform.startswith("win"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Load .env file from project root
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


@dataclass
class AppConfig:
    """Central configuration for Nexus Specialized Document Agent."""

    # Project Directories
    root_dir: Path = ROOT_DIR
    data_dir: Path = ROOT_DIR / "data"
    unstructured_dir: Path = ROOT_DIR / "data" / "unstructured"
    structured_dir: Path = ROOT_DIR / "data" / "structured"
    db_path: Path = (
        ROOT_DIR
        / os.getenv("SQLITE_DB_PATH", "data/structured/ecommerce_retail.db")
    )
    storage_dir: Path = ROOT_DIR / os.getenv("STORAGE_DIR", "storage")
    chat_history_dir: Path = ROOT_DIR / os.getenv("STORAGE_DIR", "storage") / "chat_history"

    # Model Settings (Multi-Model: Google Gemini, DeepSeek, Ollama)
    llm_provider: str = os.getenv("LLM_PROVIDER", "google").lower()
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "gemini-2.5-flash-lite")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2-preview")
    temperature: float = float(os.getenv("TEMPERATURE", "0.1"))
    agent_mode: str = os.getenv("AGENT_MODE", "function_calling")

    # DeepSeek Configuration
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    # Ollama Configuration
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
    ollama_num_gpu: int = int(os.getenv("OLLAMA_NUM_GPU", "0"))
    ollama_context_window: int = int(os.getenv("OLLAMA_CONTEXT_WINDOW", "4096"))
    enable_ollama_fallback: bool = os.getenv("ENABLE_OLLAMA_FALLBACK", "true").lower() in ("true", "1", "yes")

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
            effective = key or self.deepseek_api_key
            return bool(effective and effective.strip() != "your_deepseek_api_key_here")
        # Default to google
        effective = key or self.gemini_api_key
        return bool(
            effective
            and effective.strip() != "your_gemini_api_key_here"
        )


config = AppConfig()
config.ensure_directories()
