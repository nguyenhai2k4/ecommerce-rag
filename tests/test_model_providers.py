import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engines.model_factory import (
    initialize_models,
    check_ollama_status,
    get_available_models_for_provider,
    get_active_model_metadata,
)
from src.agent.document_agent import SpecializedDocumentAgent


def test_ollama_status_detection():
    """Verify Ollama status check returns boolean and list."""
    is_alive, models = check_ollama_status()
    assert isinstance(is_alive, bool)
    assert isinstance(models, list)
    if is_alive:
        assert len(models) >= 1
        assert "llama3.2:latest" in models or any("llama" in m.lower() for m in models)


def test_available_models_for_providers():
    """Verify curated model lists per provider."""
    google_models = get_available_models_for_provider("google")
    assert "gemini-2.5-flash-lite" in google_models

    deepseek_models = get_available_models_for_provider("deepseek")
    assert "deepseek-chat" in deepseek_models
    assert "deepseek-reasoner" in deepseek_models

    ollama_models = get_available_models_for_provider("ollama")
    assert len(ollama_models) >= 1


def test_deepseek_initialization():
    """Verify DeepSeek initializes via OpenAI-compatible endpoint without throwing exceptions."""
    llm, embed = initialize_models(
        provider="deepseek",
        api_key="sk-dummy-test-key",
        llm_model="deepseek-chat",
    )
    assert llm is not None
    meta = get_active_model_metadata()
    assert meta["provider"] == "deepseek"
    assert meta["model"] == "deepseek-chat"


def test_missing_key_fallback_to_ollama_or_mock():
    """Verify missing API key triggers fallback gracefully."""
    llm, embed = initialize_models(
        provider="google",
        api_key="",  # explicitly empty
        fallback_to_ollama=True,
    )
    assert llm is not None
    meta = get_active_model_metadata()
    assert meta["is_fallback"] is True
    # Falls back to ollama if alive, or mock
    assert meta["provider"] in ("ollama", "mock")


def test_agent_multi_model_instantiation():
    """Verify SpecializedDocumentAgent instantiates cleanly with multi-model params."""
    agent = SpecializedDocumentAgent(
        provider="deepseek",
        api_key="sk-dummy-test-key",
        llm_model="deepseek-chat",
        verbose=False,
    )
    assert agent is not None
    assert agent.provider == "deepseek"
    assert agent.metadata["provider"] == "deepseek"
    assert len(agent.tools) == 4

    summary = agent.get_vault_summary()
    assert summary["provider"] == "deepseek"
    assert summary["llm_model"] == "deepseek-chat"


if __name__ == "__main__":
    print("Running Model Provider Unit Tests...")
    test_ollama_status_detection()
    print("✓ test_ollama_status_detection passed")
    test_available_models_for_providers()
    print("✓ test_available_models_for_providers passed")
    test_deepseek_initialization()
    print("✓ test_deepseek_initialization passed")
    test_missing_key_fallback_to_ollama_or_mock()
    print("✓ test_missing_key_fallback_to_ollama_or_mock passed")
    test_agent_multi_model_instantiation()
    print("✓ test_agent_multi_model_instantiation passed")
    print("All Model Provider Tests Passed Successfully!")

