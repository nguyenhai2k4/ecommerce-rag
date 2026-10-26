from typing import Optional, List, Dict, Any
from sqlalchemy import create_engine, text, inspect
from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.core.llms import LLM
from src.config import config
from src.vault.db_seeder import seed_database


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
        self.db_path = str(db_path or config.db_path)
        seed_database()  # Ensure database exists and is seeded
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        self._refresh_sql_database()

    def _refresh_sql_database(self) -> None:
        """Inspects tables and initializes the SQLDatabase wrapper."""
        tables = self.get_table_names()
        self.sql_database = SQLDatabase(
            self.engine,
            include_tables=tables,
        )

    def get_table_names(self) -> List[str]:
        """Dynamically introspects all user tables in the database."""
        inspector = inspect(self.engine)
        return [t for t in inspector.get_table_names() if not t.startswith("sqlite_")]

    def get_foreign_key_relationships(self) -> List[str]:
        """Dynamically extracts all foreign key relationships defined across tables."""
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
        return relationships

    def get_categorical_column_guidelines(self) -> List[str]:
        """Dynamically samples distinct values for categorical/status/type columns from the live database."""
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
                                text(f"SELECT DISTINCT {col_name} FROM {table} WHERE {col_name} IS NOT NULL LIMIT 12")
                            )
                            vals = [str(r[0]) for r in res.fetchall() if r[0] is not None]
                            if vals and len(vals) <= 10:
                                formatted_vals = ", ".join(f"'{v}'" for v in vals)
                                guidelines.append(f"{table}.{col_name} values: {formatted_vals}")
                        except Exception:
                            pass
        return guidelines

    def get_schema_details(self) -> Dict[str, Any]:
        """Provides full structural metadata including columns, types, and primary/foreign keys."""
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
        return details

    def build_text_to_sql_prompt(self) -> PromptTemplate:
        """Constructs a Text-to-SQL prompt dynamically populated from database metadata."""
        relationships = self.get_foreign_key_relationships()
        guidelines = self.get_categorical_column_guidelines()
        return build_dynamic_text_to_sql_prompt(relationships=relationships, guidelines=guidelines)

    def get_query_engine(self, llm: Optional[LLM] = None) -> NLSQLTableQueryEngine:
        """Constructs an NLSQLTableQueryEngine configured with dynamically discovered schema & relationships."""
        tables = self.get_table_names()
        prompt = self.build_text_to_sql_prompt()

        return NLSQLTableQueryEngine(
            sql_database=self.sql_database,
            tables=tables,
            llm=llm,
            text_to_sql_prompt=prompt,
            verbose=True,
        )

    def execute_raw_sql(self, sql_query: str) -> List[Dict[str, Any]]:
        """Executes a read-only SQL query directly and returns row dictionaries."""
        clean_query = sql_query.strip()
        # Security check: disallow mutation queries in read-only analysis
        forbidden_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE"]
        if any(clean_query.upper().startswith(kw) for kw in forbidden_keywords):
            raise ValueError(f"Security Alert: Mutation query rejected: {clean_query[:20]}")

        with self.engine.connect() as conn:
            result = conn.execute(text(clean_query))
            keys = result.keys()
            return [dict(zip(keys, row)) for row in result.fetchall()]

    def get_table_summary(self) -> Dict[str, int]:
        """Returns row counts for each dynamically discovered table in the database."""
        summary = {}
        with self.engine.connect() as conn:
            for table in self.get_table_names():
                try:
                    res = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    summary[table] = res.scalar() or 0
                except Exception:
                    summary[table] = 0
        return summary


def _get_default_prompt() -> PromptTemplate:
    """Builds the initial default Text-to-SQL prompt dynamically from the seeded database."""
    try:
        mgr = StructuredEngineManager()
        return mgr.build_text_to_sql_prompt()
    except Exception:
        return build_dynamic_text_to_sql_prompt()


# Dynamically built prompt instance for backward compatibility
RETAIL_TEXT_TO_SQL_PROMPT = _get_default_prompt()
