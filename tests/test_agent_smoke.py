import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import config
from src.vault.db_seeder import seed_database, get_database_connection
from src.vault.doc_loader import load_vault_documents
from src.engines.structured_engine import StructuredEngineManager
from src.engines.unstructured_engine import UnstructuredEngineManager
from src.agent.document_agent import SpecializedDocumentAgent
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


if __name__ == "__main__":
    test_vault_and_database()
    test_document_loader()
    test_date_calculator_tool()
    test_dynamic_prompt_and_schema()
    test_document_management_operations()
    test_agent_initialization()
    print("All system integration tests passed successfully!")
