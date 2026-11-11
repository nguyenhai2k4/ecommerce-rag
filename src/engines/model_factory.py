import hashlib
import json
import math
import os
import re
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple
from llama_index.core import Settings
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.llms import LLM
from llama_index.core.llms.mock import MockLLM
from src.config import config


# Module-level tracking for active model metadata
_ACTIVE_METADATA: Dict[str, Any] = {
    "provider": "google",
    "model": "gemini-2.5-flash-lite",
    "is_fallback": False,
    "fallback_reason": "",
}

# Tracks whether cloud Gemini API / Embedding quota is currently exhausted (429)
_GEMINI_QUOTA_EXHAUSTED: bool = False

# Short-lived TTL cache for Ollama status checks: {url: (timestamp, is_alive, models)}
_OLLAMA_STATUS_CACHE: Dict[str, Tuple[float, bool, List[str]]] = {}
_OLLAMA_CACHE_TTL_SECONDS = 10.0

# Instance caches to avoid redundant network handshakes (e.g., GET /v1beta/models/...)
_LLM_CACHE: Dict[Tuple[Any, ...], Tuple[LLM, Dict[str, Any]]] = {}
_EMBED_CACHE: Dict[Tuple[Any, ...], BaseEmbedding] = {}

# Bounded in-memory embedding vector cache: {(model_key, text): vector}
_EMBED_VECTOR_CACHE: Dict[Tuple[str, str], List[float]] = {}
_MAX_EMBED_VECTOR_CACHE_SIZE = 2048

_CLEAN_TOKEN_RE = re.compile(r"[^a-z0-9#]+")


def mark_gemini_quota_exhausted(exhausted: bool = True) -> None:
    """Marks Gemini API quota as exhausted so subsequent calls immediately use local fallback."""
    global _GEMINI_QUOTA_EXHAUSTED
    _GEMINI_QUOTA_EXHAUSTED = exhausted


def is_gemini_quota_exhausted() -> bool:
    """Returns True if Gemini API recently returned 429 RESOURCE_EXHAUSTED."""
    return _GEMINI_QUOTA_EXHAUSTED


_SYNONYM_CANONICAL_MAP: Dict[str, str] = {
    "premier": "vip",
    "patrons": "customer",
    "buyers": "customer",
    "everyday": "standard",
    "normal": "standard",
    "dinged": "fee restocking",
    "grace": "return window",
    "reimbursement": "refund",
    "cash": "refund",
    "ripped": "opened",
    "earbuds": "in-ear earphones aerobuds",
    "cans": "headphones aerosound",
    "headset": "headphones aerosound",
    "sync": "bluetooth pairing",
    "wireless": "bluetooth",
    "restore": "reset",
    "gripes": "complaints negative feedback",
    "gym": "workout sweat",
    "client": "customer",
}


def _compute_local_semantic_vector(text: str, dim: int = 3072) -> List[float]:
    """
    Computes a deterministic, unit-normalized feature-hashed semantic vector (dim=3072)
    combining word stems, synonym canonicalization, and character n-grams.
    Used when cloud embedding APIs are offline or rate-limited (429).
    """
    cache_key = ("__local_hashed__", f"{dim}:{text}")
    if cache_key in _EMBED_VECTOR_CACHE:
        return _EMBED_VECTOR_CACHE[cache_key]

    vec = [0.0] * dim
    if not text:
        vec[0] = 1.0
        return vec

    clean = _CLEAN_TOKEN_RE.sub(" ", text.lower()).strip()
    raw_tokens = [t for t in clean.split() if len(t) > 1]
    expanded_tokens: List[str] = []
    for tok in raw_tokens:
        expanded_tokens.append(tok)
        if tok.endswith("s") and len(tok) > 3:
            expanded_tokens.append(tok[:-1])
        if tok in _SYNONYM_CANONICAL_MAP:
            expanded_tokens.extend(_SYNONYM_CANONICAL_MAP[tok].split())

    for tok in expanded_tokens:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16)
        idx = h % dim
        sign = 1.0 if (h & 1) == 0 else -1.0
        vec[idx] += sign * 2.0
        # Char 3-grams for morphological resilience
        if len(tok) >= 4:
            for i in range(len(tok) - 2):
                ng = tok[i : i + 3]
                nh = int(hashlib.md5(ng.encode("utf-8")).hexdigest()[:8], 16)
                vec[nh % dim] += (1.0 if (nh & 1) == 0 else -1.0) * 0.5

    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    else:
        vec[0] = 1.0

    if len(_EMBED_VECTOR_CACHE) < _MAX_EMBED_VECTOR_CACHE_SIZE:
        _EMBED_VECTOR_CACHE[cache_key] = vec
    return vec


class ResilientEmbedding(BaseEmbedding):
    """
    Embedding wrapper that delegates to a primary cloud/local embedding model,
    caches computed vectors in memory, and transparently falls back to deterministic
    local semantic vectors on 429/network errors.
    """
    embed_dim: int = 3072
    is_fallback: bool = False
    _primary: Optional[BaseEmbedding] = PrivateAttr(default=None)

    def __init__(
        self,
        primary: Optional[BaseEmbedding] = None,
        embed_dim: int = 3072,
        **kwargs: Any,
    ):
        super().__init__(
            model_name=getattr(primary, "model_name", "resilient-local-embedding"),
            embed_dim=embed_dim,
            is_fallback=(primary is None),
            **kwargs,
        )
        self._primary = primary

    def _cache_key(self, text: str) -> Tuple[str, str]:
        return (self.model_name, text)

    def _store_cache(self, text: str, vec: List[float]) -> List[float]:
        if len(_EMBED_VECTOR_CACHE) >= _MAX_EMBED_VECTOR_CACHE_SIZE:
            _EMBED_VECTOR_CACHE.clear()
        _EMBED_VECTOR_CACHE[self._cache_key(text)] = vec
        return vec

    def _get_query_embedding(self, query: str) -> List[float]:
        if self._primary is not None and not self.is_fallback and not _GEMINI_QUOTA_EXHAUSTED:
            ck = self._cache_key(query)
            if ck in _EMBED_VECTOR_CACHE:
                return _EMBED_VECTOR_CACHE[ck]
            try:
                return self._store_cache(query, self._primary._get_query_embedding(query))
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    mark_gemini_quota_exhausted(True)
                self.is_fallback = True
        return _compute_local_semantic_vector(query, self.embed_dim)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        if self._primary is not None and not self.is_fallback and not _GEMINI_QUOTA_EXHAUSTED:
            ck = self._cache_key(query)
            if ck in _EMBED_VECTOR_CACHE:
                return _EMBED_VECTOR_CACHE[ck]
            try:
                return self._store_cache(query, await self._primary._aget_query_embedding(query))
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    mark_gemini_quota_exhausted(True)
                self.is_fallback = True
        return _compute_local_semantic_vector(query, self.embed_dim)

    def _get_text_embedding(self, text: str) -> List[float]:
        if self._primary is not None and not self.is_fallback and not _GEMINI_QUOTA_EXHAUSTED:
            ck = self._cache_key(text)
            if ck in _EMBED_VECTOR_CACHE:
                return _EMBED_VECTOR_CACHE[ck]
            try:
                return self._store_cache(text, self._primary._get_text_embedding(text))
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    mark_gemini_quota_exhausted(True)
                self.is_fallback = True
        return _compute_local_semantic_vector(text, self.embed_dim)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        if self._primary is not None and not self.is_fallback and not _GEMINI_QUOTA_EXHAUSTED:
            ck = self._cache_key(text)
            if ck in _EMBED_VECTOR_CACHE:
                return _EMBED_VECTOR_CACHE[ck]
            try:
                return self._store_cache(text, await self._primary._aget_text_embedding(text))
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    mark_gemini_quota_exhausted(True)
                self.is_fallback = True
        return _compute_local_semantic_vector(text, self.embed_dim)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        if self._primary is not None and not self.is_fallback and not _GEMINI_QUOTA_EXHAUSTED:
            missing_indices = [i for i, t in enumerate(texts) if self._cache_key(t) not in _EMBED_VECTOR_CACHE]
            if not missing_indices:
                return [_EMBED_VECTOR_CACHE[self._cache_key(t)] for t in texts]
            try:
                missing_texts = [texts[i] for i in missing_indices]
                computed = self._primary._get_text_embeddings(missing_texts)
                for t, vec in zip(missing_texts, computed):
                    self._store_cache(t, vec)
                return [_EMBED_VECTOR_CACHE[self._cache_key(t)] for t in texts]
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    mark_gemini_quota_exhausted(True)
                self.is_fallback = True
        return [_compute_local_semantic_vector(t, self.embed_dim) for t in texts]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        if self._primary is not None and not self.is_fallback and not _GEMINI_QUOTA_EXHAUSTED:
            missing_indices = [i for i, t in enumerate(texts) if self._cache_key(t) not in _EMBED_VECTOR_CACHE]
            if not missing_indices:
                return [_EMBED_VECTOR_CACHE[self._cache_key(t)] for t in texts]
            try:
                missing_texts = [texts[i] for i in missing_indices]
                computed = await self._primary._aget_text_embeddings(missing_texts)
                for t, vec in zip(missing_texts, computed):
                    self._store_cache(t, vec)
                return [_EMBED_VECTOR_CACHE[self._cache_key(t)] for t in texts]
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    mark_gemini_quota_exhausted(True)
                self.is_fallback = True
        return [_compute_local_semantic_vector(t, self.embed_dim) for t in texts]


def get_active_model_metadata() -> Dict[str, Any]:
    """Returns the metadata of the currently initialized LLM."""
    return dict(_ACTIVE_METADATA)


def check_ollama_status(base_url: Optional[str] = None, force_refresh: bool = False) -> Tuple[bool, List[str]]:
    """
    Checks if the local or remote Ollama daemon is reachable
    and retrieves the list of installed models. Uses a short TTL cache to prevent UI lag.
    """
    url = (base_url or config.ollama_base_url or "http://localhost:11434").rstrip("/")
    now = time.monotonic()

    if not force_refresh and url in _OLLAMA_STATUS_CACHE:
        cached_ts, cached_alive, cached_models = _OLLAMA_STATUS_CACHE[url]
        if (now - cached_ts) < _OLLAMA_CACHE_TTL_SECONDS:
            return cached_alive, list(cached_models)

    try:
        req = urllib.request.Request(
            f"{url}/api/tags",
            headers={"User-Agent": "NexusAgent/1.0"},
        )
        with urllib.request.urlopen(req, timeout=1.5) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
                _OLLAMA_STATUS_CACHE[url] = (now, True, models)
                return True, list(models)
    except Exception:
        pass

    _OLLAMA_STATUS_CACHE[url] = (now, False, [])
    return False, []


def get_available_models_for_provider(provider: str, base_url: Optional[str] = None) -> List[str]:
    """Returns curated or dynamically queried model choices for the given provider."""
    p = provider.lower()
    if p in ("google", "gemini"):
        base_models = [
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
            "gemini-3.5-flash-lite",
            "gemini-1.5-pro",
            "gemini-2.0-flash-exp",
        ]
        if config.llm_model and config.llm_model not in base_models:
            base_models.insert(0, config.llm_model)
        return base_models
    elif p == "deepseek":
        base_models = [
            "deepseek-chat",
            "deepseek-reasoner",
        ]
        if config.deepseek_model and config.deepseek_model not in base_models:
            base_models.insert(0, config.deepseek_model)
        return base_models
    elif p == "ollama":
        is_alive, models = check_ollama_status(base_url)
        if is_alive and models:
            return models
        return ["llama3.2:latest", "deepseek-r1", "mistral", "qwen2.5:latest"]
    return ["mock-model"]


def _init_ollama_llm(
    target_llm: str,
    target_base_url: str,
    target_temp: float,
) -> LLM:
    """Attempts to initialize Ollama LLM with CPU-safe CUDA fallback and bounded context window."""
    from llama_index.llms.ollama import Ollama
    return Ollama(
        model=target_llm,
        base_url=target_base_url,
        temperature=target_temp,
        context_window=getattr(config, "ollama_context_window", 4096),
        request_timeout=180.0,
        additional_kwargs={
            "num_gpu": getattr(config, "ollama_num_gpu", 0),
            "num_predict": 240,
        },
    )


def _init_google_llm(
    target_llm: str,
    api_key: str,
    target_temp: float,
) -> LLM:
    """Initializes Google GenAI LLM."""
    from llama_index.llms.google_genai import GoogleGenAI
    return GoogleGenAI(
        model=target_llm,
        api_key=api_key,
        temperature=target_temp,
    )


def _init_deepseek_llm(
    target_llm: str,
    api_key: str,
    target_base_url: str,
    target_temp: float,
) -> LLM:
    """Initializes DeepSeek LLM via OpenAI-compatible endpoint."""
    try:
        from llama_index.llms.openai_like import OpenAILike
        return OpenAILike(
            model=target_llm,
            api_key=api_key,
            api_base=target_base_url,
            temperature=target_temp,
            timeout=60.0,
            is_chat_model=True,
            is_function_calling_model=True,
            context_window=64000,
        )
    except ImportError:
        from llama_index.llms.openai import OpenAI
        return OpenAI(
            model=target_llm,
            api_key=api_key,
            api_base=target_base_url,
            temperature=target_temp,
            timeout=60.0,
        )


def _resolve_embedding_model(
    preferred_provider: str,
    gemini_key: str,
    ollama_base_url: str,
    embedding_model_name: Optional[str] = None,
) -> BaseEmbedding:
    """
    Intelligently resolves an embedding model based on available credentials and providers:
    1. Google GenAI Embedding (wrapped in ResilientEmbedding) if Gemini key is available and quota is active.
    2. ResilientEmbedding deterministic local fallback (3072-dim) when running on Ollama or when cloud quota is exhausted.
    """
    clean_gemini_key = (gemini_key or "").strip()
    has_gemini_key = bool(clean_gemini_key and clean_gemini_key != "your_gemini_api_key_here")
    use_gemini_embed = has_gemini_key and not _GEMINI_QUOTA_EXHAUSTED
    embed_cache_key = (
        use_gemini_embed,
        clean_gemini_key if use_gemini_embed else "",
        ollama_base_url,
        embedding_model_name or config.embedding_model,
    )
    if embed_cache_key in _EMBED_CACHE:
        return _EMBED_CACHE[embed_cache_key]

    # 1. Try Google GenAI Embedding if key is available and provider is Google
    if use_gemini_embed:
        try:
            from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
            target_name = embedding_model_name or config.embedding_model
            primary_embed = GoogleGenAIEmbedding(
                model_name=target_name,
                api_key=clean_gemini_key,
                retries=1,
            )
            embed = ResilientEmbedding(primary=primary_embed, embed_dim=3072)
            _EMBED_CACHE[embed_cache_key] = embed
            return embed
        except Exception as e:
            print(f"[Warning] Google GenAI Embedding initialization failed: {e}")

    # 2. Fallback to ResilientEmbedding (deterministic 3072-dim semantic vector)
    embed = ResilientEmbedding(primary=None, embed_dim=3072)
    _EMBED_CACHE[embed_cache_key] = embed
    return embed


def initialize_models(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    llm_model: Optional[str] = None,
    embedding_model: Optional[str] = None,
    temperature: Optional[float] = None,
    base_url: Optional[str] = None,
    fallback_to_ollama: Optional[bool] = None,
    **kwargs: Any,
) -> Tuple[LLM, BaseEmbedding]:
    """
    Initializes and sets the global LlamaIndex LLM and Embedding models.
    Supports:
      - Google Gemini GenAI
      - DeepSeek (via OpenAI-compatible API)
      - Ollama (Local LLM daemon)
      - Seamless automatic fallback to Ollama when cloud API keys or quotas fail.
    """
    selected_provider = (provider or config.llm_provider or "google").lower()
    target_temp = temperature if temperature is not None else config.temperature
    use_ollama_fallback = (
        fallback_to_ollama
        if fallback_to_ollama is not None
        else config.enable_ollama_fallback
    )

    ollama_url = base_url or config.ollama_base_url or "http://localhost:11434"
    gemini_key = (
        api_key
        if (api_key is not None and selected_provider in ("google", "gemini"))
        else (config.gemini_api_key or os.getenv("GEMINI_API_KEY", ""))
    )
    deepseek_key = (
        api_key
        if (api_key is not None and selected_provider == "deepseek")
        else (config.deepseek_api_key or os.getenv("DEEPSEEK_API_KEY", ""))
    )

    llm_cache_key = (
        selected_provider,
        gemini_key if selected_provider in ("google", "gemini") else "",
        deepseek_key if selected_provider == "deepseek" else "",
        llm_model or "",
        target_temp,
        base_url or "",
        use_ollama_fallback,
        _GEMINI_QUOTA_EXHAUSTED,
    )

    global _ACTIVE_METADATA
    if llm_cache_key in _LLM_CACHE:
        cached_llm, cached_meta = _LLM_CACHE[llm_cache_key]
        _ACTIVE_METADATA = dict(cached_meta)
        embed_model = _resolve_embedding_model(
            preferred_provider=_ACTIVE_METADATA["provider"],
            gemini_key=gemini_key,
            ollama_base_url=ollama_url,
            embedding_model_name=embedding_model,
        )
        Settings.llm = cached_llm
        Settings.embed_model = embed_model
        return cached_llm, embed_model

    llm: Optional[LLM] = None
    active_provider = selected_provider
    active_model_name = ""
    is_fallback = False
    fallback_reason = ""

    # -------------------------------------------------------------------------
    # 1. Primary Provider: Google Gemini
    # -------------------------------------------------------------------------
    if selected_provider in ("google", "gemini"):
        target_llm = llm_model or config.llm_model or "gemini-2.5-flash-lite"
        active_model_name = target_llm
        if _GEMINI_QUOTA_EXHAUSTED and use_ollama_fallback:
            fallback_reason = "Google Gemini API quota exhausted (429)"
        elif gemini_key and gemini_key.strip() != "your_gemini_api_key_here":
            try:
                llm = _init_google_llm(target_llm, gemini_key.strip(), target_temp)
            except Exception as e:
                print(f"[Warning] Failed to initialize Google GenAI LLM ({e}).")
                fallback_reason = f"Google GenAI error: {e}"
        else:
            fallback_reason = "Google Gemini API key not configured"

    # -------------------------------------------------------------------------
    # 2. Primary Provider: DeepSeek
    # -------------------------------------------------------------------------
    elif selected_provider == "deepseek":
        target_llm = llm_model or config.deepseek_model or "deepseek-chat"
        target_base = base_url or config.deepseek_base_url or "https://api.deepseek.com/v1"
        active_model_name = target_llm
        if deepseek_key and deepseek_key.strip() != "your_deepseek_api_key_here":
            try:
                llm = _init_deepseek_llm(target_llm, deepseek_key.strip(), target_base, target_temp)
            except Exception as e:
                print(f"[Warning] Failed to initialize DeepSeek LLM ({e}).")
                fallback_reason = f"DeepSeek error: {e}"
        else:
            fallback_reason = "DeepSeek API key not configured"

    # -------------------------------------------------------------------------
    # 3. Primary Provider: Ollama (Local)
    # -------------------------------------------------------------------------
    elif selected_provider == "ollama":
        target_llm = llm_model or config.ollama_model or "llama3.2:latest"
        active_model_name = target_llm
        is_alive, available_models = check_ollama_status(ollama_url)
        if is_alive:
            # If specified model is not in tags, pick the first available one if any
            if available_models and target_llm not in available_models:
                target_llm = available_models[0]
                active_model_name = target_llm
            try:
                llm = _init_ollama_llm(target_llm, ollama_url, target_temp)
            except Exception as e:
                print(f"[Warning] Failed to initialize Ollama LLM ({e}).")
                fallback_reason = f"Ollama error: {e}"
        else:
            fallback_reason = f"Ollama daemon unreachable at {ollama_url}"

    # -------------------------------------------------------------------------
    # Fallback to Ollama if primary provider was not successfully initialized
    # -------------------------------------------------------------------------
    if llm is None and use_ollama_fallback and selected_provider != "ollama":
        is_alive, available_models = check_ollama_status(ollama_url)
        if is_alive:
            ollama_target = config.ollama_model
            if available_models and ollama_target not in available_models:
                ollama_target = available_models[0]
            try:
                llm = _init_ollama_llm(ollama_target, ollama_url, target_temp)
                active_provider = "ollama"
                active_model_name = ollama_target
                is_fallback = True
                print(f"[Info] Activated automatic fallback to local Ollama ({ollama_target}): {fallback_reason}")
            except Exception as e:
                print(f"[Warning] Fallback to Ollama failed ({e}).")

    # -------------------------------------------------------------------------
    # Final Fallback: MockLLM (for offline testing & test suites)
    # -------------------------------------------------------------------------
    if llm is None:
        llm = MockLLM(max_tokens=256)
        active_provider = "mock"
        active_model_name = "mock-llm"
        is_fallback = True
        if not fallback_reason:
            fallback_reason = "No valid model credentials found"

    # Update global active metadata safely without modifying strict Pydantic models
    _ACTIVE_METADATA = {
        "provider": active_provider,
        "model": active_model_name,
        "is_fallback": is_fallback,
        "fallback_reason": fallback_reason,
    }
    _LLM_CACHE[llm_cache_key] = (llm, dict(_ACTIVE_METADATA))

    # Best-effort attribute setting if object allows (e.g. MockLLM or custom wrappers)
    try:
        setattr(llm, "provider_name", active_provider)
        setattr(llm, "active_model_name", active_model_name)
        setattr(llm, "is_fallback", is_fallback)
        setattr(llm, "fallback_reason", fallback_reason)
    except Exception:
        pass

    # Resolve Embeddings
    embed_model = _resolve_embedding_model(
        preferred_provider=active_provider,
        gemini_key=gemini_key,
        ollama_base_url=ollama_url,
        embedding_model_name=embedding_model,
    )

    Settings.llm = llm
    Settings.embed_model = embed_model

    return llm, embed_model
