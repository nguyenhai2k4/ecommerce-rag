"""
Arize Phoenix OpenTelemetry & OpenInference Instrumentation for LlamaIndex.

Provides:
  - 1-click LlamaIndex automatic instrumentation (`LlamaIndexInstrumentor`)
  - Runtime compatibility patch for OpenInference + Google GenAI usage_metadata lists
  - Local Arize Phoenix server lifecycle management (`http://localhost:6006`)
  - Custom OpenInference-compatible spans for custom multi-hop agent steps and RAG evaluations
"""

import logging
import os
import socket
import time
import urllib.parse
import urllib.request
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple

from src.config import config, configure_clean_logging

_PHOENIX_INSTRUMENTED: bool = False
_PHOENIX_SESSION: Any = None
_TRACER: Any = None
_HANDLER_PATCHED: bool = False

# Short TTL cache for Phoenix server reachability: {base_url: (timestamp, is_running)}
_PHOENIX_STATUS_CACHE: Dict[str, Tuple[float, bool]] = {}
_PHOENIX_CACHE_TTL_SECONDS: float = 10.0


def _patch_openinference_handler() -> None:
    """
    Patches `openinference.instrumentation.llama_index._handler` so that
    `_get_token_counts_from_mapping` safely handles non-dict subfields such as
    Google GenAI's `prompt_tokens_details` (which is a list of ModalityTokenCount
    dicts rather than an OpenAI-style dict). Prevents `AttributeError: 'list' object
    has no attribute 'get'` on `LLMChatEndEvent` and allows Google GenAI token
    counts to be recorded properly.
    """
    global _HANDLER_PATCHED
    if _HANDLER_PATCHED:
        return
    try:
        import openinference.instrumentation.llama_index._handler as _oi_handler

        logging.getLogger("openinference.instrumentation.llama_index").setLevel(logging.CRITICAL)
        logging.getLogger("openinference.instrumentation.llama_index._handler").setLevel(logging.CRITICAL)

        impl_fn = getattr(_oi_handler, "_get_token_counts_impl", None)
        if callable(impl_fn):
            def _safe_get_token_counts_from_mapping(
                usage_mapping: Mapping[str, Any],
            ) -> Iterator[Tuple[str, Any]]:
                def get_value(obj: Any, key: str) -> Any:
                    if isinstance(obj, Mapping):
                        return obj.get(key)
                    if isinstance(obj, (list, tuple)):
                        return None
                    return getattr(obj, key, None)

                yield from impl_fn(usage_mapping, get_value)

            _oi_handler._get_token_counts_from_mapping = _safe_get_token_counts_from_mapping

        _HANDLER_PATCHED = True
    except Exception:
        pass


_patch_openinference_handler()


def is_phoenix_server_running(endpoint: Optional[str] = None, force_refresh: bool = False) -> bool:
    """
    Checks if an Arize Phoenix server is reachable at the configured HTTP endpoint.
    Uses a fast TCP socket pre-check and a short TTL cache to avoid blocking turns when offline.
    """
    base_url = (endpoint or config.phoenix_base_url or "http://localhost:6006").rstrip("/")
    now = time.monotonic()

    if not force_refresh and base_url in _PHOENIX_STATUS_CACHE:
        cached_ts, cached_up = _PHOENIX_STATUS_CACHE[base_url]
        if (now - cached_ts) < _PHOENIX_CACHE_TTL_SECONDS:
            return cached_up

    try:
        parsed = urllib.parse.urlparse(base_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 6006)
        # Fast socket check (250ms) avoids 1.5s+ Windows localhost SYN retry delays when port is closed
        with socket.create_connection((host, port), timeout=0.25):
            pass

        req = urllib.request.Request(base_url, headers={"User-Agent": "NexusAgent-PhoenixCheck/1.0"})
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            is_up = resp.status in (200, 301, 302, 307, 308)
            _PHOENIX_STATUS_CACHE[base_url] = (now, is_up)
            return is_up
    except Exception:
        _PHOENIX_STATUS_CACHE[base_url] = (now, False)
        return False


def launch_local_phoenix_server(port: int = 6006) -> Tuple[bool, str]:
    """
    Launches a local Arize Phoenix UI server in the background if not already running.
    Returns (success, url_or_error_message).
    """
    global _PHOENIX_SESSION
    url = f"http://localhost:{port}"
    if is_phoenix_server_running(url, force_refresh=True):
        setup_phoenix_tracing(endpoint=f"{url}/v1/traces", force=True)
        return True, url

    try:
        import phoenix as px
        os.environ.setdefault("PHOENIX_PORT", str(port))
        _PHOENIX_SESSION = px.launch_app(use_temp_dir=False)
        active_url = (getattr(_PHOENIX_SESSION, "url", url) or url).rstrip("/")
        _PHOENIX_STATUS_CACHE[active_url] = (time.monotonic(), True)
        setup_phoenix_tracing(endpoint=f"{active_url}/v1/traces", force=True)
        return True, active_url
    except Exception as e:
        return False, f"Failed to launch Phoenix server: {e}"


def setup_phoenix_tracing(
    endpoint: Optional[str] = None,
    project_name: Optional[str] = None,
    force: bool = False,
) -> bool:
    """
    Registers OpenTelemetry tracer provider with Arize Phoenix and instruments LlamaIndex.
    Safe to call multiple times (idempotent unless force=True).
    """
    global _PHOENIX_INSTRUMENTED, _TRACER

    _patch_openinference_handler()

    if _PHOENIX_INSTRUMENTED and not force:
        return True

    if not config.enable_phoenix_tracing and not force:
        return False

    target_base = (config.phoenix_base_url or "http://localhost:6006").rstrip("/")
    target_endpoint = endpoint or config.phoenix_collector_endpoint or f"{target_base}/v1/traces"
    target_project = project_name or config.phoenix_project_name or "nexus-document-agent"

    # Only attach network exporter if Phoenix collector is reachable or cloud API key is configured
    server_up = is_phoenix_server_running(target_base, force_refresh=force)
    if not server_up and not os.getenv("PHOENIX_API_KEY"):
        return False

    try:
        from phoenix.otel import register
        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
        from opentelemetry import trace

        _patch_openinference_handler()

        tracer_provider = register(
            project_name=target_project,
            endpoint=target_endpoint,
            set_global_tracer_provider=True,
        )
        instrumentor = LlamaIndexInstrumentor()
        if not getattr(instrumentor, "_is_instrumented_by_opentelemetry", False):
            instrumentor.instrument(tracer_provider=tracer_provider)

        configure_clean_logging()
        _TRACER = trace.get_tracer("nexus.document.agent")
        _PHOENIX_INSTRUMENTED = True
        return True
    except Exception as e:
        print(f"[Phoenix] Tracing initialization skipped: {e}")
        return False


def get_phoenix_status(force_refresh: bool = False) -> Dict[str, Any]:
    """Returns current Arize Phoenix connection and instrumentation status for UI display."""
    base_url = (config.phoenix_base_url or "http://localhost:6006").rstrip("/")
    server_up = is_phoenix_server_running(base_url, force_refresh=force_refresh)
    if server_up and not _PHOENIX_INSTRUMENTED and config.enable_phoenix_tracing:
        setup_phoenix_tracing()

    return {
        "server_running": server_up,
        "instrumented": _PHOENIX_INSTRUMENTED,
        "base_url": base_url,
        "collector_endpoint": config.phoenix_collector_endpoint or f"{base_url}/v1/traces",
        "project_name": config.phoenix_project_name,
    }


@contextmanager
def trace_agent_turn(
    user_query: str,
    provider: str = "",
    model_name: str = "",
):
    """
    Context manager that wraps an agent turn in an OpenInference AGENT span
    when Arize Phoenix tracing is active.
    """
    if not _PHOENIX_INSTRUMENTED and config.enable_phoenix_tracing:
        setup_phoenix_tracing()

    if not _PHOENIX_INSTRUMENTED or _TRACER is None:
        yield None
        return

    try:
        with _TRACER.start_as_current_span("SpecializedDocumentAgent.chat") as span:
            span.set_attribute("openinference.span.kind", "AGENT")
            span.set_attribute("input.value", user_query)
            span.set_attribute("llm.provider", provider)
            span.set_attribute("llm.model_name", model_name)
            yield span
    except Exception:
        yield None


def record_agent_span_result(
    span: Any,
    response_text: str,
    sources: List[str],
    retrieved_contexts: List[str],
    latency_seconds: float,
    trace_steps: Optional[List[Any]] = None,
) -> None:
    """Attaches final answer, citations, retrieved context chunks, and tool calls to the active Phoenix span."""
    if span is None:
        return
    try:
        span.set_attribute("output.value", response_text)
        span.set_attribute("agent.latency_seconds", latency_seconds)
        span.set_attribute("agent.sources", ", ".join(sources))
        span.set_attribute("agent.retrieved_context_count", len(retrieved_contexts))
        for idx, ctx in enumerate(retrieved_contexts[:5]):
            span.set_attribute(f"retrieval.documents.{idx}.document.content", ctx[:1000])
        if trace_steps:
            for idx, step in enumerate(trace_steps):
                t_name = getattr(step, "tool_name", None) or getattr(step, "step_type", "step")
                t_in = getattr(step, "input_data", "") or ""
                t_out = getattr(step, "output_data", "") or ""
                span.set_attribute(f"agent.steps.{idx}.tool_name", str(t_name))
                span.set_attribute(f"agent.steps.{idx}.input", str(t_in)[:400])
                span.set_attribute(f"agent.steps.{idx}.output", str(t_out)[:400])
    except Exception:
        pass


def record_evaluation_span(
    case_id: str,
    query: str,
    response: str,
    metrics: Dict[str, Any],
) -> None:
    """Logs a LlamaIndex Built-in Evaluator result as an EVALUATOR span in Arize Phoenix."""
    if not _PHOENIX_INSTRUMENTED and config.enable_phoenix_tracing:
        setup_phoenix_tracing()

    if not _PHOENIX_INSTRUMENTED or _TRACER is None:
        return

    try:
        with _TRACER.start_as_current_span(f"LlamaIndexEval.{case_id}") as span:
            span.set_attribute("openinference.span.kind", "EVALUATOR")
            span.set_attribute("input.value", query)
            span.set_attribute("output.value", response)
            span.set_attribute("eval.case_id", case_id)
            for k, v in metrics.items():
                if isinstance(v, (int, float, bool, str)):
                    span.set_attribute(f"eval.{k}", v)
    except Exception:
        pass
