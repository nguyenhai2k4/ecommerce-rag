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

    # Model Settings (Google Gemini / GenAI)
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "gemini-3.8-flash")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2-preview")
    temperature: float = float(os.getenv("TEMPERATURE", "0.1"))
    agent_mode: str = os.getenv("AGENT_MODE", "function_calling")

    def ensure_directories(self) -> None:
        """Create necessary directories if they do not exist."""
        self.unstructured_dir.mkdir(parents=True, exist_ok=True)
        self.structured_dir.mkdir(parents=True, exist_ok=True)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def is_api_key_configured(self) -> bool:
        """Check if a valid Gemini API key is configured."""
        return bool(
            self.gemini_api_key
            and self.gemini_api_key != "your_gemini_api_key_here"
        )


config = AppConfig()
config.ensure_directories()
