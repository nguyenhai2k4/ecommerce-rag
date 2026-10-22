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

__all__ = [
    "SpecializedDocumentAgent",
    "AgentResponse",
    "AgentTraceStep",
    "DOCUMENT_AGENT_SYSTEM_PROMPT",
    "build_document_agent_tools",
]
