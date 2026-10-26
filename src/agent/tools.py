from datetime import datetime
from typing import List, Optional
from llama_index.core.tools import QueryEngineTool, FunctionTool, BaseTool
from src.engines.structured_engine import StructuredEngineManager
from src.engines.unstructured_engine import UnstructuredEngineManager


def calculate_days_between_dates(date_start: str, date_end: str = "2026-10-22") -> int:
    """
    Calculates the exact number of calendar days between two dates formatted as YYYY-MM-DD.
    Useful for verifying return window eligibility and shipping delivery SLAs.

    Args:
        date_start: Start date in 'YYYY-MM-DD' format (e.g., delivery_date).
        date_end: End date in 'YYYY-MM-DD' format (defaults to current date '2026-10-22').

    Returns:
        The integer number of calendar days elapsed.
    """
    try:
        d1 = datetime.strptime(date_start.strip(), "%Y-%m-%d")
        d2 = datetime.strptime(date_end.strip(), "%Y-%m-%d")
        return abs((d2 - d1).days)
    except Exception as e:
        return f"Error calculating date difference: {e}"


def build_document_agent_tools(
    structured_manager: StructuredEngineManager,
    unstructured_manager: UnstructuredEngineManager,
    llm=None,
) -> List[BaseTool]:
    """
    Constructs the tool suite for the Specialized Document Agent.
    """
    # 1. SQL Query Tool
    tables = structured_manager.get_table_names()
    tables_str = ", ".join(tables) if tables else "products, customers, orders, inventory_logs"
    sql_engine = structured_manager.get_query_engine(llm=llm)
    sql_tool = QueryEngineTool.from_defaults(
        query_engine=sql_engine,
        name="retail_sql_analytics",
        description=(
            "Translates natural language questions into SQL to query structured data from the retail database. "
            f"Use this tool whenever you need exact counts, metrics, orders, delivery dates, prices, customer loyalty tiers, "
            f"stock quantities, or inventory logs. Dynamically available tables: {tables_str}."
        ),
    )

    # 2. Vector Semantic Search Tool
    from src.vault.doc_loader import load_vault_documents
    active_docs = load_vault_documents()
    unique_titles = list({d.metadata.get("title", d.metadata.get("file_name", "Doc")) for d in active_docs})
    docs_str = "; ".join(unique_titles) if unique_titles else "corporate policy, SLA, specification, and sentiment documents"

    vector_engine = unstructured_manager.get_vector_query_engine(similarity_top_k=3)
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_engine,
        name="knowledge_vault_semantic_search",
        description=(
            "Performs semantic search across the unstructured knowledge vault. "
            f"Active indexed documents: {docs_str}. "
            "Covers return & refund policies, tier windows (Standard 30d, Gold 45d, VIP 60d), shipping SLAs, "
            "product technical specifications, and customer review sentiment."
        ),
    )

    # 3. Document Summary Tool
    summary_engine = unstructured_manager.get_summary_query_engine()
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_engine,
        name="document_summary_tool",
        description=(
            "Synthesizes high-level overviews across full documents. "
            "Use this tool when asked for broad summaries, comparisons of entire policies, or general document overviews."
        ),
    )

    # 4. Precision Date Utility Tool
    date_tool = FunctionTool.from_defaults(
        fn=calculate_days_between_dates,
        name="date_calculator",
        description=(
            "Calculates exact calendar days between two YYYY-MM-DD dates (e.g., delivery_date and current date). "
            "Prevents arithmetic hallucinations when verifying if an order falls within a return policy window."
        ),
    )

    return [sql_tool, vector_tool, summary_tool, date_tool]
