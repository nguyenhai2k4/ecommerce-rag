import os
from typing import Optional, Tuple
from llama_index.core import Settings
from llama_index.core.llms import LLM
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms.mock import MockLLM
from llama_index.core.embeddings import MockEmbedding
from src.config import config


def initialize_models(
    api_key: Optional[str] = None,
    llm_model: Optional[str] = None,
    embedding_model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> Tuple[LLM, BaseEmbedding]:
    """
    Initializes and sets the global LlamaIndex LLM and Embedding models.
    Supports Google Gemini GenAI with graceful mock fallback for testing without credentials.
    """
    effective_api_key = api_key or config.gemini_api_key or os.getenv("GEMINI_API_KEY", "")
    target_llm = llm_model or config.llm_model
    target_embed = embedding_model or config.embedding_model
    target_temp = temperature if temperature is not None else config.temperature

    if effective_api_key and effective_api_key.strip() != "your_gemini_api_key_here":
        try:
            from llama_index.llms.google_genai import GoogleGenAI
            from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

            llm = GoogleGenAI(
                model=target_llm,
                api_key=effective_api_key,
                temperature=target_temp,
            )
            embed_model = GoogleGenAIEmbedding(
                model_name=target_embed,
                api_key=effective_api_key,
            )
            Settings.llm = llm
            Settings.embed_model = embed_model
            return llm, embed_model
        except Exception as e:
            print(f"[Warning] Failed to initialize Google GenAI models ({e}). Falling back to mock models.")

    # Mock fallback for offline verification & tests
    mock_llm = MockLLM(max_tokens=256)
    mock_embed = MockEmbedding(embed_dim=768)
    Settings.llm = mock_llm
    Settings.embed_model = mock_embed
    return mock_llm, mock_embed
