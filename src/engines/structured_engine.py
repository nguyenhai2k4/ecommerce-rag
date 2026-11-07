import re
from pathlib import Path
from typing import Optional, List, Dict, Any
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine
from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.core.llms import LLM
from src.config import config
from src.vault.db_seeder import seed_database

# Shared SQLAlchemy engine & metadata cache per resolved db_path
_ENGINE_CACHE: Dict[str, Engine] = {}
_SCHEMA_CACHE: Dict[str, Dict[str, Any]] = {}


def build_dynamic_text_to_sql_prompt(
    relationships: Optional[List[str]] = None,
    guidelines: Optional[List[str]] = None,
) -> PromptTemplate:
    """Builds a fully dynamic Text-to-SQL prompt without hardcoded schema, tables, or categories."""
    rel_section = (
        "Table Relationships:\n" + "\n".join(f"- {r}" for r in relationships)
        if relationships
        else "Table Relationships:\n- Refer to foreign keys declared in the schema information above."
    )

    guide_lines = []
    if guidelines:
        guide_lines.extend(f"- {g}" for g in guidelines)
    guide_lines.append("- For dates, use standard SQLite ISO format 'YYYY-MM-DD' or strftime where appropriate.")
    guide_section = "Important Column Guidelines:\n" + "\n".join(guide_lines)

    template_str = f"""Given an input question, first create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.

You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the few relevant columns given the question.
Pay attention to use only the column names that you can see in the schema description. Be careful not to query for columns that do not exist.
Pay attention to which column is in which table. Also, qualify column names with table names when joining.

Schema Information:
{{schema}}

{rel_section}

{guide_section}

Question: {{query_str}}
SQLQuery: """
    return PromptTemplate(template_str)


class StructuredEngineManager:
    """Manages dynamic Text-to-SQL query generation over the structured SQLite database."""

    def __init__(self, db_path: Optional[str] = None):
        resolved_path = Path(db_path or config.db_path).resolve()
        self.db_path = str(resolved_path)
        seed_database(resolved_path)  # Ensure database exists and is seeded

        if self.db_path not in _ENGINE_CACHE:
            _ENGINE_CACHE[self.db_path] = create_engine(f"sqlite:///{self.db_path}")
        self.engine = _ENGINE_CACHE[self.db_path]
        self._refresh_sql_database()

    def _get_cache_bucket(self) -> Dict[str, Any]:
        """Returns or initializes the schema cache bucket for this database path and file mtime."""
        try:
            mtime = Path(self.db_path).stat().st_mtime_ns
        except OSError:
            mtime = 0
        bucket = _SCHEMA_CACHE.get(self.db_path)
        if bucket is None or bucket.get("mtime") != mtime:
            bucket = {"mtime": mtime}
            _SCHEMA_CACHE[self.db_path] = bucket
        return bucket

    def _refresh_sql_database(self) -> None:
        """Inspects tables and initializes the SQLDatabase wrapper."""
        bucket = self._get_cache_bucket()
        if "sql_database" not in bucket:
            tables = self.get_table_names()
            bucket["sql_database"] = SQLDatabase(
                self.engine,
                include_tables=tables,
            )
        self.sql_database = bucket["sql_database"]

    def get_table_names(self) -> List[str]:
        """Dynamically introspects all user tables in the database."""
        bucket = self._get_cache_bucket()
        if "table_names" not in bucket:
            inspector = inspect(self.engine)
            bucket["table_names"] = [t for t in inspector.get_table_names() if not t.startswith("sqlite_")]
        return list(bucket["table_names"])

    def get_foreign_key_relationships(self) -> List[str]:
        """Dynamically extracts all foreign key relationships defined across tables."""
        bucket = self._get_cache_bucket()
        if "fk_relationships" not in bucket:
            inspector = inspect(self.engine)
            relationships = []
            for table in self.get_table_names():
                fks = inspector.get_foreign_keys(table)
                for fk in fks:
                    referred_table = fk.get("referred_table")
                    constrained_cols = fk.get("constrained_columns", [])
                    referred_cols = fk.get("referred_columns", [])
                    for c_col, r_col in zip(constrained_cols, referred_cols):
                        relationships.append(f"{table}.{c_col} joins with {referred_table}.{r_col}")
            bucket["fk_relationships"] = relationships
        return list(bucket["fk_relationships"])

    def get_categorical_column_guidelines(self) -> List[str]:
        """Dynamically samples distinct values for categorical/status/type columns from the live database."""
        bucket = self._get_cache_bucket()
        if "guidelines" not in bucket:
            guidelines = []
            inspector = inspect(self.engine)
            with self.engine.connect() as conn:
                for table in self.get_table_names():
                    for col in inspector.get_columns(table):
                        col_name = col["name"]
                        # Identify categorical/enumerated columns dynamically
                        if any(term in col_name.lower() for term in ["tier", "status", "category", "event_type", "type"]):
                            try:
                                res = conn.execute(
                                    text(f'SELECT DISTINCT "{col_name}" FROM "{table}" WHERE "{col_name}" IS NOT NULL LIMIT 12')
                                )
                                vals = [str(r[0]) for r in res.fetchall() if r[0] is not None]
                                if vals and len(vals) <= 10:
                                    formatted_vals = ", ".join(f"'{v}'" for v in vals)
                                    guidelines.append(f"{table}.{col_name} values: {formatted_vals}")
                            except Exception:
                                pass
            bucket["guidelines"] = guidelines
        return list(bucket["guidelines"])

    def get_schema_details(self) -> Dict[str, Any]:
        """Provides full structural metadata including columns, types, and primary/foreign keys."""
        bucket = self._get_cache_bucket()
        if "schema_details" not in bucket:
            inspector = inspect(self.engine)
            details = {}
            for table in self.get_table_names():
                columns = inspector.get_columns(table)
                pk = inspector.get_pk_constraint(table).get("constrained_columns", [])
                fks = inspector.get_foreign_keys(table)
                details[table] = {
                    "columns": [
                        {
                            "name": col["name"],
                            "type": str(col["type"]),
                            "is_pk": col["name"] in pk,
                            "nullable": col.get("nullable", True),
                        }
                        for col in columns
                    ],
                    "primary_keys": pk,
                    "foreign_keys": fks,
                }
            bucket["schema_details"] = details
        return dict(bucket["schema_details"])

    def build_text_to_sql_prompt(self) -> PromptTemplate:
        """Constructs a Text-to-SQL prompt dynamically populated from database metadata."""
        bucket = self._get_cache_bucket()
        if "prompt" not in bucket:
            relationships = self.get_foreign_key_relationships()
            guidelines = self.get_categorical_column_guidelines()
            bucket["prompt"] = build_dynamic_text_to_sql_prompt(relationships=relationships, guidelines=guidelines)
        return bucket["prompt"]

    def get_query_engine(self, llm: Optional[LLM] = None, verbose: bool = False) -> NLSQLTableQueryEngine:
        """Constructs an NLSQLTableQueryEngine configured with dynamically discovered schema & relationships."""
        tables = self.get_table_names()
        prompt = self.build_text_to_sql_prompt()

        return NLSQLTableQueryEngine(
            sql_database=self.sql_database,
            tables=tables,
            llm=llm,
            text_to_sql_prompt=prompt,
            verbose=verbose,
        )

    def execute_raw_sql(self, sql_query: str) -> List[Dict[str, Any]]:
        """Executes a read-only SQL query directly and returns row dictionaries."""
        clean_query = sql_query.strip()
        # Strip SQL comments before checking keywords
        stripped_no_comments = re.sub(r"(--[^\n]*|/\*.*?\*/)", "", clean_query, flags=re.DOTALL).strip()
        forbidden_keywords = ("DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "REPLACE", "ATTACH", "DETACH", "PRAGMA")
        upper_q = stripped_no_comments.upper()
        if any(upper_q.startswith(kw) or re.search(rf"\b{kw}\b", upper_q) for kw in forbidden_keywords):
            raise ValueError(f"Security Alert: Mutation query rejected: {clean_query[:40]}")

        with self.engine.connect() as conn:
            result = conn.execute(text(clean_query))
            keys = result.keys()
            return [dict(zip(keys, row)) for row in result.fetchall()]

    def get_table_summary(self) -> Dict[str, int]:
        """Returns row counts for each dynamically discovered table in the database."""
        bucket = self._get_cache_bucket()
        if "table_summary" not in bucket:
            summary = {}
            with self.engine.connect() as conn:
                for table in self.get_table_names():
                    try:
                        res = conn.execute(text(f'SELECT COUNT(*) FROM "{table}"'))
                        summary[table] = res.scalar() or 0
                    except Exception:
                        summary[table] = 0
            bucket["table_summary"] = summary
        return dict(bucket["table_summary"])


# Lightweight default prompt instance for backward compatibility (avoids import-time DB I/O)
RETAIL_TEXT_TO_SQL_PROMPT = build_dynamic_text_to_sql_prompt()
