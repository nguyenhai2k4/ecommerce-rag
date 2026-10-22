import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from llama_index.core.agent import AgentWorkflow, FunctionAgent, ReActAgent
from llama_index.core.llms import LLM
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from src.config import config
from src.engines.model_factory import initialize_models
from src.engines.structured_engine import StructuredEngineManager
from src.engines.unstructured_engine import UnstructuredEngineManager
from src.agent.prompts import DOCUMENT_AGENT_SYSTEM_PROMPT
from src.agent.tools import build_document_agent_tools


@dataclass
class AgentTraceStep:
    """Represents a single step in the agent's reasoning trace."""
    step_type: str  # 'thought', 'tool_call', 'tool_result', 'final_answer'
    tool_name: Optional[str] = None
    input_data: Optional[str] = None
    output_data: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentResponse:
    """Encapsulates the final answer, reasoning steps, tool usage, and performance latency."""
    response_text: str
    sources: List[str] = field(default_factory=list)
    trace_steps: List[AgentTraceStep] = field(default_factory=list)
    latency_seconds: float = 0.0
    model_name: str = ""


class SpecializedDocumentAgent:
    """
    Enterprise-grade Document & Data Intelligence Agent.
    Orchestrates multi-hop queries across structured SQL databases
    and unstructured vector knowledge vaults.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        llm: Optional[LLM] = None,
        verbose: bool = True,
    ):
        self.api_key = api_key or config.gemini_api_key
        self.verbose = verbose

        # 1. Initialize models
        if llm is not None:
            self.llm = llm
        else:
            self.llm, _ = initialize_models(api_key=self.api_key)

        # 2. Initialize knowledge engines
        self.structured_manager = StructuredEngineManager()
        self.unstructured_manager = UnstructuredEngineManager()

        # 3. Build agent tools
        self.tools = build_document_agent_tools(
            structured_manager=self.structured_manager,
            unstructured_manager=self.unstructured_manager,
            llm=self.llm,
        )

        # 4. Construct Agent Workflow
        self.workflow = AgentWorkflow.from_tools_or_functions(
            tools_or_functions=self.tools,
            llm=self.llm,
            system_prompt=DOCUMENT_AGENT_SYSTEM_PROMPT,
            verbose=self.verbose,
        )

        # 5. Conversation History
        self.chat_history: List[ChatMessage] = []

    def get_vault_summary(self) -> Dict[str, Any]:
        """Provides statistical overview of both structured and unstructured vaults."""
        table_counts = self.structured_manager.get_table_summary()
        docs = self.unstructured_manager.retrieve_nodes("policy specifications SLA review", similarity_top_k=10)
        doc_titles = list({node.node.metadata.get("title", "Unknown") for node in docs})

        return {
            "structured_tables": table_counts,
            "unstructured_documents": doc_titles,
            "llm_model": config.llm_model,
            "embedding_model": config.embedding_model,
        }

    async def achat(self, user_query: str) -> AgentResponse:
        """Asynchronous execution of the agent workflow with trace instrumentation."""
        start_time = time.time()
        trace_steps: List[AgentTraceStep] = []

        # If API key is not configured, return clear guidance
        if not config.is_api_key_configured() and not self.api_key:
            return self._generate_api_key_required_response(user_query, start_time)

        try:
            # Run the agent workflow
            response_obj = await self.workflow.run(user_msg=user_query)
            final_text = getattr(response_obj, "response", str(response_obj))

            # Deduce sources referenced
            sources = []
            if "POL-RET-2026-V3" in final_text or "Return" in final_text or "Policy" in final_text:
                sources.append("Return, Refund & Warranty Policy (POL-RET-2026-V3)")
            if "SLA-LOG-2026-V2" in final_text or "Shipping" in final_text or "SLA" in final_text:
                sources.append("Shipping & Fulfillment SLA (SLA-LOG-2026-V2)")
            if "SPEC-TECH-2026-A1" in final_text or "Specification" in final_text or "AeroSound" in final_text:
                sources.append("Product Specifications & Troubleshooting Guide (SPEC-TECH-2026-A1)")
            if "CSAT-REP-2026-Q3" in final_text or "Review" in final_text or "Sentiment" in final_text:
                sources.append("Customer Feedback & Sentiment Report Q3 (CSAT-REP-2026-Q3)")
            if "order" in user_query.lower() or "stock" in user_query.lower() or "price" in user_query.lower() or "customer" in user_query.lower():
                sources.append("Retail SQLite Database (tables: products, orders, customers)")

            # Record in conversation history
            self.chat_history.append(ChatMessage(role=MessageRole.USER, content=user_query))
            self.chat_history.append(ChatMessage(role=MessageRole.ASSISTANT, content=str(final_text)))

            latency = round(time.time() - start_time, 2)
            return AgentResponse(
                response_text=str(final_text),
                sources=list(set(sources)),
                trace_steps=trace_steps,
                latency_seconds=latency,
                model_name=config.llm_model,
            )

        except Exception as e:
            latency = round(time.time() - start_time, 2)
            return AgentResponse(
                response_text=f"An error occurred during agent reasoning: {str(e)}\n\nPlease ensure your `GEMINI_API_KEY` is valid and has sufficient quota.",
                sources=[],
                trace_steps=[AgentTraceStep(step_type="error", output_data=str(e))],
                latency_seconds=latency,
                model_name=config.llm_model,
            )

    def chat(self, user_query: str) -> AgentResponse:
        """Synchronous wrapper for chat execution."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(self.achat(user_query))
            else:
                return asyncio.run(self.achat(user_query))
        except RuntimeError:
            return asyncio.run(self.achat(user_query))

    def reset_conversation(self) -> None:
        """Clears chat history and restarts agent session."""
        self.chat_history.clear()

    def _generate_api_key_required_response(self, user_query: str, start_time: float) -> AgentResponse:
        """Provides informative response when GEMINI_API_KEY is not yet supplied."""
        demo_analysis = (
            "⚠️ **Gemini API Key Required**\n\n"
            "To execute live inference with Gemini 2.5 Flash and text-embedding-004, please configure your `GEMINI_API_KEY`:\n\n"
            "1. Copy `.env.example` to `.env`:\n"
            "   ```bash\n"
            "   cp .env.example .env\n"
            "   ```\n"
            "2. Add your Google AI Studio key: `GEMINI_API_KEY=AIzaSy...`\n"
            "3. Or enter your key in the Streamlit web app sidebar.\n\n"
            "---\n"
            "### Planned Multi-Hop Reasoning Plan for Query:\n"
            f"> *\"{user_query}\"*\n\n"
            "- **Step 1 (Structured Engine)**: Query SQLite DB tables (`products`, `customers`, `orders`) for IDs, dates, quantities, and tiers.\n"
            "- **Step 2 (Unstructured Engine)**: Semantic search in Knowledge Vault for policy rules, specs, and sentiment.\n"
            "- **Step 3 (Cross-Domain Synthesis)**: Validate policy limits (e.g., 30 vs 60-day return window) against order delivery dates and synthesize final grounded answer."
        )
        return AgentResponse(
            response_text=demo_analysis,
            sources=["System Configuration Guide"],
            trace_steps=[AgentTraceStep(step_type="notice", output_data="API Key Missing")],
            latency_seconds=round(time.time() - start_time, 2),
            model_name="Configuration Required",
        )
