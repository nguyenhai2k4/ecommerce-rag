import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import config
from src.vault.db_seeder import seed_database, get_database_connection
from src.vault.doc_loader import load_vault_documents
from src.engines.structured_engine import StructuredEngineManager
from src.engines.unstructured_engine import UnstructuredEngineManager
from src.agent.document_agent import SpecializedDocumentAgent
from src.agent.chat_history import ChatHistoryManager, auto_generate_title
from src.agent.tools import calculate_days_between_dates


def test_vault_and_database():
    """Verify SQLite database seeding and row counts."""
    db_path = seed_database(force=True)
    assert db_path.exists()

    mgr = StructuredEngineManager(db_path=db_path)
    summary = mgr.get_table_summary()
    assert summary["products"] == 6
    assert summary["customers"] == 5
    assert summary["orders"] == 6
    assert summary["inventory_logs"] == 4


def test_document_loader():
    """Verify unstructured markdown document loading and metadata enrichment."""
    docs = load_vault_documents()
    assert len(docs) >= 4
    titles = [d.metadata.get("title") for d in docs]
    assert "Return, Refund & Warranty Policy" in titles
    assert "Shipping & Fulfillment SLA" in titles


def test_date_calculator_tool():
    """Verify deterministic date calculation tool."""
    days = calculate_days_between_dates("2026-09-10", "2026-10-22")
    assert days == 42


def test_agent_initialization():
    """Verify agent initializes tools and handles query planning."""
    agent = SpecializedDocumentAgent(verbose=False)
    assert len(agent.tools) == 4
    res = agent.chat("Can Alice Chen return her order?")
    assert res is not None
    assert len(res.response_text) > 0


def test_dynamic_prompt_and_schema():
    """Verify dynamic schema extraction, foreign key discovery, and dynamic system prompt generation."""
    from src.agent.prompts import (
        extract_database_schema_and_fks,
        extract_vault_manifest,
        build_dynamic_system_prompt,
    )

    db_info = extract_database_schema_and_fks()
    assert "products" in db_info["table_names"]
    assert "orders" in db_info["table_names"]
    assert len(db_info["fk_relationships"]) >= 3
    assert any("orders.customer_id joins with customers.customer_id" in rel for rel in db_info["fk_relationships"])

    vault_info = extract_vault_manifest()
    assert len(vault_info["entries"]) >= 4

    agent = SpecializedDocumentAgent(verbose=False)
    assert "Live Database Schema" in agent.system_prompt
    assert "orders.customer_id joins with customers.customer_id" in agent.system_prompt
    assert "retail_sql_analytics" in agent.system_prompt
    assert "knowledge_vault_semantic_search" in agent.system_prompt


def test_document_management_operations():
    """Verify adding, previewing, and deleting documents with auto-reindexing."""
    mgr = UnstructuredEngineManager()
    test_filename = "_temp_smoke_test_policy.md"
    test_content = b"# Temporary Smoke Test Policy\nThis document verifies dynamic upload, indexing, and deletion."

    # 1. Add document
    file_path = mgr.add_document(test_filename, test_content)
    assert file_path.exists()

    vault_files = mgr.get_vault_file_list()
    assert any(f["name"] == test_filename for f in vault_files)

    # 2. Verify agent reload picks it up
    agent = SpecializedDocumentAgent(verbose=False)
    agent.reload_vault(force_reindex=False)
    assert test_filename in agent.system_prompt or "Temporary Smoke Test Policy" in agent.system_prompt

    # 3. Delete document
    deleted = mgr.delete_document(test_filename)
    assert deleted is True
    assert not file_path.exists()
    assert not any(f["name"] == test_filename for f in mgr.get_vault_file_list())


def test_chat_history_and_session_lifecycle():
    """Verify chat session persistence, auto-titling, markdown export, and memory syncing."""
    history_mgr = ChatHistoryManager()
    
    # 1. Title Auto-Generation
    title1 = auto_generate_title("Can you please check if Alice Chen can return order #5001?")
    assert "Alice" in title1
    assert "return" in title1.lower()

    # 2. Create Session
    session = history_mgr.create_session("Initial Title")
    session_id = session["id"]
    assert session_id.startswith("chat_")
    assert len(session["messages"]) == 1
    assert session["messages"][0]["role"] == "assistant"

    # 3. List Sessions & Initialization Filtering
    sessions = history_mgr.list_sessions(require_user_messages=False)
    assert any(s["id"] == session_id for s in sessions)
    # Since session only has assistant greeting, it is not yet initialized with user messages
    uninitialized_sessions = history_mgr.list_sessions(require_user_messages=True)
    assert not any(s["id"] == session_id for s in uninitialized_sessions)

    # 4. Save and Update Conversation with auto-titling (now initialized)
    updated_messages = [
        {"role": "assistant", "content": "Welcome greeting"},
        {"role": "user", "content": "How do I factory reset AeroSound headphones?"},
        {"role": "assistant", "content": "Hold the power button for 10 seconds.", "sources": ["SPEC-TECH-2026-A1"], "latency": 1.2},
    ]
    saved = history_mgr.save_session(
        session_id=session_id,
        title="New Conversation",  # Should trigger auto-titling from user query
        messages=updated_messages,
    )
    assert "AeroSound" in saved["title"] or "reset" in saved["title"].lower()
    assert len(saved["messages"]) == 3
    # Now it must appear in initialized sessions
    initialized_sessions = history_mgr.list_sessions(require_user_messages=True)
    assert any(s["id"] == session_id for s in initialized_sessions)

    # 5. Agent Memory Loading & Resetting
    agent = SpecializedDocumentAgent(verbose=False)
    agent.load_conversation(updated_messages)
    assert len(agent.chat_history) == 3
    assert agent.chat_history[1].content == "How do I factory reset AeroSound headphones?"
    agent.reset_conversation()
    assert len(agent.chat_history) == 0

    # 6. Export Session Markdown
    md_export = history_mgr.export_session_markdown(session_id)
    assert "AeroSound" in md_export
    assert "SPEC-TECH-2026-A1" in md_export

    # 7. Rename Session
    renamed = history_mgr.rename_session(session_id, "Custom Handcrafted Title")
    assert renamed is True
    loaded_renamed = history_mgr.get_session(session_id)
    assert loaded_renamed["title"] == "Custom Handcrafted Title"

    # 8. Delete Session
    deleted = history_mgr.delete_session(session_id)
    assert deleted is True
    assert history_mgr.get_session(session_id) is None

    # 9. Cleanup uninitialized sessions
    ghost_sess = history_mgr.create_session("Ghost Session")
    assert history_mgr.get_session(ghost_sess["id"]) is not None
    cleaned_count = history_mgr.cleanup_uninitialized_sessions()
    assert cleaned_count >= 1
    assert history_mgr.get_session(ghost_sess["id"]) is None


def test_openinference_google_genai_token_counts_patch():
    """Verify OpenInference handler safely processes Google GenAI usage_metadata where prompt_tokens_details is a list."""
    from src.evaluation.phoenix_tracing import _patch_openinference_handler
    import openinference.instrumentation.llama_index._handler as oi_handler

    _patch_openinference_handler()
    sample_google_usage = {
        "prompt_token_count": 180,
        "candidates_token_count": 45,
        "total_token_count": 225,
        "prompt_tokens_details": [{"modality": "TEXT", "token_count": 180}],
        "candidates_tokens_details": [{"modality": "TEXT", "token_count": 45}],
    }
    extracted = dict(oi_handler._get_token_counts(sample_google_usage))
    assert extracted.get("llm.token_count.prompt") == 180
    assert extracted.get("llm.token_count.completion") == 45
    assert extracted.get("llm.token_count.total") == 225


if __name__ == "__main__":
    test_vault_and_database()
    test_document_loader()
    test_date_calculator_tool()
    test_dynamic_prompt_and_schema()
    test_document_management_operations()
    test_chat_history_and_session_lifecycle()
    test_openinference_google_genai_token_counts_patch()
    test_agent_initialization()
    print("All system integration tests passed successfully!")
