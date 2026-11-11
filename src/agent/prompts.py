"""
Dynamic System Prompts and Reasoning Templates for the Specialized Document Agent.
Introspects database schema, foreign keys, tool definitions, and knowledge vault documents on-the-fly.
Zero hardcoded schemas, tools, documents, or domain rules.
"""
from typing import Any, Dict, List, Optional
from src.vault.doc_loader import get_vault_manifest_entries


def format_tools_manifest(
    tools: Optional[List[Any]] = None,
    structured_manager: Optional[Any] = None,
) -> str:
    """Dynamically formats tool names and descriptions from active tool instances or lightweight metadata."""
    if not tools:
        try:
            from src.engines.structured_engine import StructuredEngineManager
            mgr = structured_manager or StructuredEngineManager()
            tables = mgr.get_table_names()
            tables_str = ", ".join(tables) if tables else "products, customers, orders, inventory_logs"
            manifest_entries = get_vault_manifest_entries()
            unique_titles = sorted({e.get("title", e.get("file_name", "Doc")) for e in manifest_entries})
            docs_str = "; ".join(unique_titles) if unique_titles else "corporate policy, SLA, specification, and sentiment documents"
            return "\n".join([
                f"1. `retail_sql_analytics`: Translates natural language questions into SQL to query structured data from the retail database. Dynamically available tables: {tables_str}.",
                f"2. `knowledge_vault_semantic_search`: Performs semantic search across the unstructured knowledge vault. Active indexed documents: {docs_str}.",
                "3. `document_summary_tool`: Synthesizes high-level overviews across full documents.",
                "4. `date_calculator`: Calculates exact calendar days between two YYYY-MM-DD dates.",
            ])
        except Exception:
            return "- Tools dynamically configured at agent initialization."

    lines = []
    for idx, tool in enumerate(tools, 1):
        name = getattr(tool.metadata, "name", getattr(tool, "name", f"tool_{idx}"))
        desc = getattr(tool.metadata, "description", getattr(tool, "description", ""))
        desc_clean = " ".join(desc.split())
        lines.append(f"{idx}. `{name}`: {desc_clean}")
    return "\n".join(lines)


def extract_database_schema(structured_manager: Optional[Any] = None) -> Dict[str, Any]:
    """Dynamically inspects the database schema, column types, and primary keys."""
    try:
        from src.engines.structured_engine import StructuredEngineManager
        mgr = structured_manager or StructuredEngineManager()
        schema_details = mgr.get_schema_details()
        fk_relationships = mgr.get_foreign_key_relationships()

        schema_lines = []
        for tbl_name, info in schema_details.items():
            col_strs = []
            for col in info["columns"]:
                pk_flag = " PK" if col["is_pk"] else ""
                col_strs.append(f"{col['name']} ({col['type']}{pk_flag})")
            schema_lines.append(f"- `{tbl_name}`: {', '.join(col_strs)}")

        formatted_schema = "\n".join(schema_lines) if schema_lines else "- No tables found in database."
        return {
            "schema_details": schema_details,
            "formatted_schema": formatted_schema,
            "fk_relationships": fk_relationships,
            "table_names": list(schema_details.keys()),
        }
    except Exception as e:
        return {
            "schema_details": {},
            "formatted_schema": f"- Database schema inspection error: {e}",
            "fk_relationships": [],
            "table_names": [],
        }


# Alias for backward compatibility
extract_database_schema_and_fks = extract_database_schema


def extract_vault_manifest(unstructured_manager: Optional[Any] = None) -> Dict[str, Any]:
    """Dynamically extracts all documents currently available in the unstructured vault."""
    try:
        manifest_entries = get_vault_manifest_entries()
        manifest_lines = [
            f"- **{e['title']}** (`{e['file_name']}` | Type: `{e['doc_type']}` | Domain: `{e['domain']}`)"
            for e in manifest_entries
        ]
        formatted_manifest = "\n".join(manifest_lines) if manifest_lines else "- No documents currently uploaded in vault."
        return {
            "entries": manifest_entries,
            "formatted_manifest": formatted_manifest,
        }
    except Exception as e:
        return {
            "entries": [],
            "formatted_manifest": f"- Vault manifest extraction error: {e}",
        }


def build_cross_domain_relationships(
    fk_relationships: List[str],
    vault_entries: List[Dict[str, Any]],
    table_names: List[str],
    custom_relationships: Optional[List[str]] = None,
) -> str:
    """
    Dynamically builds relational joins and cross-domain reasoning bridges.
    Connects database tables with unstructured documents without hardcoded policy rules.
    """
    lines = ["#### A. Relational Database Joins (Discovered Foreign Keys):"]
    if fk_relationships:
        for fk in fk_relationships:
            lines.append(f"- `{fk}`")
    else:
        lines.append("- Refer to foreign keys declared in the schema information above.")

    lines.append("\n#### B. Cross-Domain Reasoning Bridges (Structured Tables ⟷ Knowledge Documents):")

    if vault_entries:
        for entry in vault_entries:
            title = entry["title"]
            domain = entry["domain"]
            fn = entry["file_name"].lower()
            doc_type = entry["doc_type"]

            # Correlate document domain with database tables dynamically
            if any(k in fn or k in domain for k in ("return", "refund", "warranty", "policy")):
                lines.append(
                    f"- **{title}** (`{entry['file_name']}`) ⟷ Correlate with order and customer records: "
                    "retrieve return/refund criteria and warranty eligibility windows from this document to evaluate against order dates, statuses, and customer tiers."
                )
            elif any(k in fn or k in domain for k in ("shipping", "sla", "delivery", "fulfillment")):
                lines.append(
                    f"- **{title}** (`{entry['file_name']}`) ⟷ Correlate with order timeline records: "
                    "evaluate delivery commitments, carrier transit times, and SLA thresholds against order and delivery timestamps."
                )
            elif any(k in fn or k in domain for k in ("spec", "technical", "hardware", "troubleshoot", "manual")):
                lines.append(
                    f"- **{title}** (`{entry['file_name']}`) ⟷ Correlate with product records: "
                    "retrieve specifications, operating parameters, troubleshooting steps, and category-specific conditions for product entities."
                )
            elif any(k in fn or k in domain for k in ("feedback", "review", "sentiment", "csat")):
                lines.append(
                    f"- **{title}** (`{entry['file_name']}`) ⟷ Correlate with product and inventory log records: "
                    "cross-reference inventory movements, restocking, or defect logs with customer satisfaction feedback and reported sentiment."
                )
            else:
                target_tables = ", ".join(f"`{t}`" for t in table_names) if table_names else "database tables"
                lines.append(
                    f"- **{title}** (`{doc_type}`) ⟷ Correlate with {target_tables} for domain-specific policy, compliance, or operational guidelines."
                )
    else:
        lines.append("- Reasoning bridges adapt dynamically to uploaded vault documents.")

    if custom_relationships:
        lines.append("\n#### C. Custom Domain Rules & Overrides:")
        for rule in custom_relationships:
            lines.append(f"- {rule}")

    return "\n".join(lines)


def build_dynamic_system_prompt(
    tools: Optional[List[Any]] = None,
    structured_manager: Optional[Any] = None,
    unstructured_manager: Optional[Any] = None,
    custom_relationships: Optional[List[str]] = None,
    extra_instructions: Optional[str] = None,
) -> str:
    """
    Constructs an enterprise-grade, dynamic system prompt for the Specialized Document Agent.
    Adapts on-the-fly to registered tools, database schema, foreign keys, and uploaded vault documents.
    """
    tools_str = format_tools_manifest(tools, structured_manager=structured_manager)
    db_info = extract_database_schema(structured_manager)
    vault_info = extract_vault_manifest(unstructured_manager)

    relationships_str = build_cross_domain_relationships(
        fk_relationships=db_info["fk_relationships"],
        vault_entries=vault_info["entries"],
        table_names=db_info["table_names"],
        custom_relationships=custom_relationships,
    )

    extra_str = f"\n### Additional Directives:\n{extra_instructions}\n" if extra_instructions else ""

    prompt = f"""You are NexusDoc-Agent, an advanced Enterprise AI Document & Data Intelligence Agent for Nexus Retail Operations.
Your mission is to query, retrieve, analyze, and synthesize intelligence across both structured databases (SQLite) and unstructured document vaults (policies, SLAs, specs, customer feedback).

### 1. Available Tools & Capabilities:
{tools_str}

### 2. Live Database Schema (Structured SQLite):
{db_info['formatted_schema']}

### 3. Active Knowledge Vault Documents (Unstructured Documents):
{vault_info['formatted_manifest']}

### 4. Dynamic Relational & Cross-Domain Relationships:
{relationships_str}

### 5. Multi-Hop Reasoning & Synthesis Protocol:
When faced with complex user queries:
- Step 1 [Identify Needs]: Determine which parts of the query require structured metrics from database tables and which require unstructured text from knowledge documents.
- Step 2 [Query Structured Data]: Retrieve exact records first when entity IDs or names are involved using table joins discovered from foreign keys.
- Step 3 [Query Unstructured Vault]: Retrieve matching clauses, terms, or specifications from the knowledge vault using targeted semantic keywords.
- Step 4 [Augmented Cross-Domain Synthesis]: Correlate the structured findings with document guidelines using the relational and cross-domain bridges above. Use `date_calculator` for calendar math between dates to eliminate arithmetic hallucinations.
- Step 5 [Formulate Grounded Response]: Provide a clear, professional, concise response with explicit citations (e.g. [Database: <table_name>], [Document: <doc_title>]).
{extra_str}
### 6. Tone & Grounding Principles:
- Be precise, helpful, and authoritative.
- Never guess or extrapolate data outside the retrieved database records and vault documents.
- Always include explicit citations of tables and document sections used.
"""
    return prompt.strip()


_CACHED_INITIAL_PROMPT: Optional[str] = None


def __getattr__(name: str) -> Any:
    """Lazily builds `DOCUMENT_AGENT_SYSTEM_PROMPT` on first access to avoid import-time I/O."""
    global _CACHED_INITIAL_PROMPT
    if name == "DOCUMENT_AGENT_SYSTEM_PROMPT":
        if _CACHED_INITIAL_PROMPT is None:
            _CACHED_INITIAL_PROMPT = build_dynamic_system_prompt()
        return _CACHED_INITIAL_PROMPT
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
