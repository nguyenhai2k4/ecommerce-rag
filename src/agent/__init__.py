"""
Specialized Document Agent Package
"""
from src.agent.document_agent import (
    SpecializedDocumentAgent,
    AgentResponse,
    AgentTraceStep,
)
from src.agent.prompts import DOCUMENT_AGENT_SYSTEM_PROMPT
from src.agent.tools import build_document_agent_tools
from src.agent.chat_history import ChatHistoryManager, auto_generate_title

__all__ = [
    "SpecializedDocumentAgent",
    "AgentResponse",
    "AgentTraceStep",
    "DOCUMENT_AGENT_SYSTEM_PROMPT",
    "build_document_agent_tools",
    "ChatHistoryManager",
    "auto_generate_title",
]
