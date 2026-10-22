"""
Knowledge Vault Management (Structured SQLite & Unstructured Ingestion)
"""
from src.vault.db_seeder import seed_database, get_database_connection
from src.vault.doc_loader import load_vault_documents

__all__ = ["seed_database", "get_database_connection", "load_vault_documents"]
