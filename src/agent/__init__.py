"""
Specialized Document Agent Package
"""
from typing import Any
from src.agent.chat_history import ChatHistoryManager, auto_generate_title
from src.agent.document_agent import (
    AgentResponse,
    AgentTraceStep,
    SpecializedDocumentAgent,
)
from src.agent.tools import build_document_agent_tools

__all__ = [
    "SpecializedDocumentAgent",
    "AgentResponse",
    "AgentTraceStep",
    "DOCUMENT_AGENT_SYSTEM_PROMPT",
    "build_document_agent_tools",
    "ChatHistoryManager",
    "auto_generate_title",
]


def __getattr__(name: str) -> Any:
    if name == "DOCUMENT_AGENT_SYSTEM_PROMPT":
        from src.agent import prompts
        return getattr(prompts, "DOCUMENT_AGENT_SYSTEM_PROMPT")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
