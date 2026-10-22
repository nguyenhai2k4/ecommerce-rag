from typing import Optional, List, Dict, Any
from sqlalchemy import create_engine, text
from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.core.llms import LLM
from src.config import config
from src.vault.db_seeder import seed_database


RETAIL_TEXT_TO_SQL_PROMPT = PromptTemplate(
    """Given an input question, first create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.

You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the few relevant columns given the question.
Pay attention to use only the column names that you can see in the schema description. Be careful not to query for columns that do not exist.
Pay attention to which column is in which table. Also, qualify column names with table names when joining.

Schema Information:
{schema}

Table Relationships:
- orders.customer_id joins with customers.customer_id
- orders.product_id joins with products.product_id
- inventory_logs.product_id joins with products.product_id

Important Guidelines:
- Customer tiers: 'Standard', 'Gold', 'VIP'
- Product categories: 'Audio', 'Peripherals', 'Wearables', 'Power', 'Accessories'
- Order statuses: 'Delivered', 'In Transit', 'Processing', 'Cancelled', 'Returned'
- For dates, use standard SQLite ISO format 'YYYY-MM-DD' or strftime where appropriate.

Question: {query_str}
SQLQuery: """
)


class StructuredEngineManager:
    """Manages Text-to-SQL query generation over the structured SQLite database."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path or config.db_path)
        seed_database()  # Ensure database exists and is seeded
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        self.sql_database = SQLDatabase(
            self.engine,
            include_tables=["products", "customers", "orders", "inventory_logs"],
        )

    def get_query_engine(self, llm: Optional[LLM] = None) -> NLSQLTableQueryEngine:
        """Constructs an NLSQLTableQueryEngine configured with custom schema context."""
        return NLSQLTableQueryEngine(
            sql_database=self.sql_database,
            tables=["products", "customers", "orders", "inventory_logs"],
            llm=llm,
            text_to_sql_prompt=RETAIL_TEXT_TO_SQL_PROMPT,
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
        """Returns row counts for each table in the database."""
        summary = {}
        with self.engine.connect() as conn:
            for table in ["products", "customers", "orders", "inventory_logs"]:
                res = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                summary[table] = res.scalar() or 0
        return summary
