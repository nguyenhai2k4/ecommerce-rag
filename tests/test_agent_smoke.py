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


if __name__ == "__main__":
    test_vault_and_database()
    test_document_loader()
    test_date_calculator_tool()
    test_agent_initialization()
    print("All system integration tests passed successfully!")
