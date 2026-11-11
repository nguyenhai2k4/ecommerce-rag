import math
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from rich.console import Console
from rich.table import Table

import sqlglot
from sqlglot import exp

from src.config import config, configure_clean_logging


@dataclass
class SQLTestCase:
    """Benchmark test case for Text-to-SQL evaluation."""
    id: str
    prompt: str
    gold_sql: str
    category: str  # "Simple", "Joins", "Aggregations", "Complex"
    expected_row_count: int
    description: str
    ordered: bool = False  # Set to True if order matters (e.g. ORDER BY ... LIMIT)


# =============================================================================
# 20 Calibrated Golden Benchmark Test Cases for Nexus Retail Schema
# =============================================================================
SQL_BENCHMARK_SUITE: List[SQLTestCase] = [
    # Tier 1: Simple Filter & Projection (Easy)
    SQLTestCase(
        id="SQL-01",
        prompt="List the names and prices of all active products in the Audio category.",
        gold_sql="SELECT name, price FROM products WHERE category = 'Audio' AND status = 'Active';",
        category="Simple",
        expected_row_count=2,
        description="Filter on multiple attributes with AND condition.",
    ),
    SQLTestCase(
        id="SQL-02",
        prompt="What are the names and prices of the top 3 most expensive products?",
        gold_sql="SELECT name, price FROM products ORDER BY price DESC LIMIT 3;",
        category="Simple",
        expected_row_count=3,
        description="Projection with ORDER BY descending and LIMIT.",
        ordered=True,
    ),
    SQLTestCase(
        id="SQL-03",
        prompt="How many customers are registered under the VIP tier?",
        gold_sql="SELECT COUNT(*) AS vip_count FROM customers WHERE tier = 'VIP';",
        category="Simple",
        expected_row_count=1,
        description="Scalar COUNT aggregate with WHERE filter.",
    ),
    SQLTestCase(
        id="SQL-04",
        prompt="Find all orders that do not have a delivery date recorded yet.",
        gold_sql="SELECT order_id, customer_id, status FROM orders WHERE delivery_date IS NULL;",
        category="Simple",
        expected_row_count=1,
        description="IS NULL check on optional timestamp column.",
    ),
    SQLTestCase(
        id="SQL-05",
        prompt="List all unique cities where customers are registered, in alphabetical order.",
        gold_sql="SELECT DISTINCT city FROM customers ORDER BY city ASC;",
        category="Simple",
        expected_row_count=5,
        description="DISTINCT values with ascending alphabetization.",
        ordered=True,
    ),

    # Tier 2: Relational Joins & Lookups (Medium)
    SQLTestCase(
        id="SQL-06",
        prompt="Retrieve the order ID, order date, and total amount for all orders placed by Alice Chen.",
        gold_sql="""
            SELECT o.order_id, o.order_date, o.total_amount 
            FROM orders o 
            JOIN customers c ON o.customer_id = c.customer_id 
            WHERE c.name = 'Alice Chen';
        """,
        category="Joins",
        expected_row_count=2,
        description="Two-table INNER JOIN on customer_id with customer name filter.",
    ),
    SQLTestCase(
        id="SQL-07",
        prompt="Which product name and quantity were purchased in order ID 5001?",
        gold_sql="""
            SELECT p.name, o.quantity 
            FROM orders o 
            JOIN products p ON o.product_id = p.product_id 
            WHERE o.order_id = 5001;
        """,
        category="Joins",
        expected_row_count=1,
        description="Two-table INNER JOIN on product_id with primary key filter.",
    ),
    SQLTestCase(
        id="SQL-08",
        prompt="What is the combined total spend across all orders placed by VIP customers?",
        gold_sql="""
            SELECT SUM(o.total_amount) AS vip_total_spend 
            FROM orders o 
            JOIN customers c ON o.customer_id = c.customer_id 
            WHERE c.tier = 'VIP';
        """,
        category="Joins",
        expected_row_count=1,
        description="JOIN with aggregate SUM over joined customer tier.",
    ),
    SQLTestCase(
        id="SQL-09",
        prompt="Show the log ID, product name, units, and timestamp for all restock events of the AeroSound Pro Wireless ANC Headphones.",
        gold_sql="""
            SELECT l.log_id, p.name, l.units, l.logged_at 
            FROM inventory_logs l 
            JOIN products p ON l.product_id = p.product_id 
            WHERE p.name = 'AeroSound Pro Wireless ANC Headphones' AND l.event_type = 'Restock';
        """,
        category="Joins",
        expected_row_count=1,
        description="JOIN on inventory_logs with compound filter on product name and event type.",
    ),
    SQLTestCase(
        id="SQL-10",
        prompt="List the customer name, product name, and order date for all Delivered orders.",
        gold_sql="""
            SELECT c.name AS customer_name, p.name AS product_name, o.order_date 
            FROM orders o 
            JOIN customers c ON o.customer_id = c.customer_id 
            JOIN products p ON o.product_id = p.product_id 
            WHERE o.status = 'Delivered';
        """,
        category="Joins",
        expected_row_count=5,
        description="Three-table JOIN connecting customers, orders, and products.",
    ),

    # Tier 3: Aggregations, GROUP BY & HAVING (Hard)
    SQLTestCase(
        id="SQL-11",
        prompt="How many total orders have been placed per customer tier? Show tier and count, sorted highest first.",
        gold_sql="""
            SELECT c.tier, COUNT(o.order_id) AS order_count 
            FROM customers c 
            LEFT JOIN orders o ON c.customer_id = o.customer_id 
            GROUP BY c.tier 
            ORDER BY order_count DESC;
        """,
        category="Aggregations",
        expected_row_count=3,
        description="GROUP BY on joined category with sorting.",
        ordered=True,
    ),
    SQLTestCase(
        id="SQL-12",
        prompt="Calculate total sales revenue generated by each product category, ordered from highest to lowest revenue.",
        gold_sql="""
            SELECT p.category, SUM(o.total_amount) AS category_revenue 
            FROM orders o 
            JOIN products p ON o.product_id = p.product_id 
            GROUP BY p.category 
            ORDER BY category_revenue DESC;
        """,
        category="Aggregations",
        expected_row_count=3,
        description="GROUP BY category with SUM aggregate and descending order.",
        ordered=True,
    ),
    SQLTestCase(
        id="SQL-13",
        prompt="Which customers have placed more than 1 order? Return customer name and order count.",
        gold_sql="""
            SELECT c.name, COUNT(o.order_id) AS order_count 
            FROM customers c 
            JOIN orders o ON c.customer_id = o.customer_id 
            GROUP BY c.customer_id, c.name 
            HAVING COUNT(o.order_id) > 1;
        """,
        category="Aggregations",
        expected_row_count=1,
        description="GROUP BY with HAVING filter on aggregated count.",
    ),
    SQLTestCase(
        id="SQL-14",
        prompt="What is the average price of products in each category, rounded to 2 decimal places?",
        gold_sql="""
            SELECT category, ROUND(AVG(price), 2) AS avg_price 
            FROM products 
            GROUP BY category 
            ORDER BY avg_price DESC;
        """,
        category="Aggregations",
        expected_row_count=5,
        description="Single table GROUP BY with ROUND and AVG functions.",
        ordered=True,
    ),
    SQLTestCase(
        id="SQL-15",
        prompt="What is the total number of product units restocked across all inventory logs?",
        gold_sql="""
            SELECT SUM(units) AS total_units_restocked 
            FROM inventory_logs 
            WHERE event_type = 'Restock';
        """,
        category="Aggregations",
        expected_row_count=1,
        description="SUM aggregate on inventory logs with event type filter.",
    ),

    # Tier 4: Subqueries, Dates & Multi-Condition Logic (Extra Hard)
    SQLTestCase(
        id="SQL-16",
        prompt="Which products have never been ordered? Return their names.",
        gold_sql="""
            SELECT name 
            FROM products 
            WHERE product_id NOT IN (SELECT DISTINCT product_id FROM orders);
        """,
        category="Complex",
        expected_row_count=2,
        description="Anti-join via NOT IN subquery.",
    ),
    SQLTestCase(
        id="SQL-17",
        prompt="Find the order ID, customer ID, and total amount of the single highest value order.",
        gold_sql="""
            SELECT order_id, customer_id, total_amount 
            FROM orders 
            ORDER BY total_amount DESC 
            LIMIT 1;
        """,
        category="Complex",
        expected_row_count=1,
        description="Single extreme value retrieval with ORDER BY and LIMIT 1.",
        ordered=True,
    ),
    SQLTestCase(
        id="SQL-18",
        prompt="List the order ID, order date, and total amount for all orders placed in October 2026.",
        gold_sql="""
            SELECT order_id, order_date, total_amount 
            FROM orders 
            WHERE strftime('%Y-%m', order_date) = '2026-10';
        """,
        category="Complex",
        expected_row_count=4,
        description="Date formatting and month isolation using SQLite strftime.",
    ),
    SQLTestCase(
        id="SQL-19",
        prompt="Calculate the turnaround duration in days between order date and delivery date for each delivered order.",
        gold_sql="""
            SELECT order_id, CAST(julianday(delivery_date) - julianday(order_date) AS INTEGER) AS delivery_days 
            FROM orders 
            WHERE status = 'Delivered';
        """,
        category="Complex",
        expected_row_count=5,
        description="Date math in SQLite via julianday.",
    ),
    SQLTestCase(
        id="SQL-20",
        prompt="For every customer, display their name, tier, total order count, and total amount spent (showing 0.0 if they have no orders), sorted by total spent descending.",
        gold_sql="""
            SELECT c.name, c.tier, COUNT(o.order_id) AS total_orders, COALESCE(SUM(o.total_amount), 0.0) AS total_spent 
            FROM customers c 
            LEFT JOIN orders o ON c.customer_id = o.customer_id 
            GROUP BY c.customer_id, c.name, c.tier 
            ORDER BY total_spent DESC;
        """,
        category="Complex",
        expected_row_count=5,
        description="LEFT JOIN, multiple grouping keys, COALESCE fallback for NULL, and ORDER BY.",
        ordered=True,
    ),
]


@dataclass
class SQLTestResult:
    """Outcome of testing a single SQL query."""
    test_id: str
    category: str
    prompt: str
    generated_sql: str
    gold_sql: str
    canonical_gen_sql: str
    canonical_gold_sql: str
    exact_ast_match: bool
    component_matches: Dict[str, bool]
    exec_accuracy: bool
    exec_success: bool
    latency_ms: float
    error: Optional[str] = None
    gen_rows: Optional[List[Any]] = None
    gold_rows: Optional[List[Any]] = None


class SQLASTEvaluator:
    """
    Production-grade Abstract Syntax Tree (AST) evaluator for Text-to-SQL queries.
    Features:
      - AST parsing and safety verification (blocks mutating / dangerous operations)
      - AST canonicalization (normalizes aliases, identifiers, commutative AND/OR logic)
      - Fine-grained component matching (SELECT, FROM, WHERE, GROUP BY, HAVING, ORDER BY, LIMIT)
      - Safe, read-only SQLite execution with deterministic comparison
    """

    def __init__(self, db_path: Optional[Union[Path, str]] = None):
        self.db_path = Path(db_path or config.db_path).resolve()
        self.console = Console()

    @staticmethod
    def extract_sql(text: str) -> str:
        """Extract clean SQL string from LLM responses, stripping fences, prefixes, and comments."""
        if not text:
            return ""
        # Match ```sql ... ``` or ``` ... ```
        fence_match = re.search(r"```(?:sql)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        if fence_match:
            text = fence_match.group(1)
        # Strip common prefixes like 'SQLQuery:' or 'SQL:'
        text = re.sub(r"^(?:SQLQuery|SQL):\s*", "", text.strip(), flags=re.IGNORECASE)
        # Remove comments
        text = re.sub(r"(--[^\n]*|/\*.*?\*/)", "", text)
        return text.strip().rstrip(";")

    @staticmethod
    def verify_safety(ast_tree: exp.Expression) -> Tuple[bool, Optional[str]]:
        """Verifies that the AST only contains read-only SELECT / UNION operations."""
        forbidden_types = (
            exp.Insert,
            exp.Update,
            exp.Delete,
            exp.Drop,
            exp.Alter,
            exp.Command,
            exp.Pragma,
        )
        for bad_cls in forbidden_types:
            if isinstance(ast_tree, bad_cls) or ast_tree.find(bad_cls):
                return False, f"Prohibited SQL statement detected: {bad_cls.__name__}"
        return True, None

    @classmethod
    def canonicalize_ast(cls, sql_str: str) -> Tuple[Optional[exp.Expression], Optional[str]]:
        """
        Parses SQL and transforms the AST into a canonical form:
          - Replaces arbitrary table aliases with deterministic t1, t2, ...
          - Recursively sorts commutative operators (AND / OR)
          - Lowercases identifiers
        """
        clean_sql = cls.extract_sql(sql_str)
        if not clean_sql:
            return None, "Empty SQL query"

        try:
            tree = sqlglot.parse_one(clean_sql, read="sqlite")
        except Exception as e:
            return None, f"SQL Syntax/Parsing Error: {e}"

        # 1. Deterministic alias normalization
        tables = [t.name.lower() for t in tree.find_all(exp.Table) if t.name]
        unique_tables = list(dict.fromkeys(tables))
        table_to_canonical = {tbl: f"t{i+1}" for i, tbl in enumerate(unique_tables)}

        alias_map = {}
        for t in tree.find_all(exp.Table):
            tbl_name = t.name.lower() if t.name else ""
            canon = table_to_canonical.get(tbl_name, tbl_name)
            if t.alias:
                alias_map[t.alias.lower()] = canon
            t.set("alias", exp.TableAlias(this=exp.to_identifier(canon)))

        for col in tree.find_all(exp.Column):
            if col.table:
                old_tbl = col.table.lower()
                if old_tbl in alias_map:
                    col.set("table", exp.to_identifier(alias_map[old_tbl]))
                elif old_tbl in table_to_canonical:
                    col.set("table", exp.to_identifier(table_to_canonical[old_tbl]))

        # 2. Reorder commutative binary expressions (AND, OR)
        def sort_commutative(node):
            if isinstance(node, (exp.And, exp.Or)):
                operands = []
                curr = node
                while isinstance(curr, type(node)):
                    operands.append(curr.right)
                    curr = curr.left
                operands.append(curr)

                norm_operands = [sort_commutative(op) for op in operands]
                norm_operands.sort(key=lambda x: x.sql(dialect="sqlite").lower())

                res = norm_operands[0]
                op_cls = type(node)
                for op in norm_operands[1:]:
                    res = op_cls(this=res, expression=op)
                return res
            return node

        canonical_tree = tree.transform(sort_commutative)
        return canonical_tree, None

    @classmethod
    def compare_components(
        cls,
        gen_tree: Optional[exp.Expression],
        gold_tree: Optional[exp.Expression],
    ) -> Dict[str, bool]:
        """Performs clause-by-clause AST component comparison."""
        results = {
            "select": False,
            "from": False,
            "where": False,
            "group_by": False,
            "having": False,
            "order_by": False,
            "limit": False,
        }
        if not gen_tree or not gold_tree:
            return results

        # 1. SELECT expressions (as unordered set of normalized SQL strings)
        gen_sel = {e.sql(dialect="sqlite").lower() for e in getattr(gen_tree, "expressions", [])}
        gold_sel = {e.sql(dialect="sqlite").lower() for e in getattr(gold_tree, "expressions", [])}
        results["select"] = (gen_sel == gold_sel)

        # 2. FROM & JOIN
        gen_from = (gen_tree.find(exp.From).sql(dialect="sqlite").lower()) if gen_tree.find(exp.From) else ""
        gold_from = (gold_tree.find(exp.From).sql(dialect="sqlite").lower()) if gold_tree.find(exp.From) else ""
        gen_joins = sorted([j.sql(dialect="sqlite").lower() for j in gen_tree.find_all(exp.Join)])
        gold_joins = sorted([j.sql(dialect="sqlite").lower() for j in gold_tree.find_all(exp.Join)])
        results["from"] = (gen_from == gold_from) and (gen_joins == gold_joins)

        # 3. WHERE clause
        gen_where = (gen_tree.find(exp.Where).sql(dialect="sqlite").lower()) if gen_tree.find(exp.Where) else ""
        gold_where = (gold_tree.find(exp.Where).sql(dialect="sqlite").lower()) if gold_tree.find(exp.Where) else ""
        results["where"] = (gen_where == gold_where)

        # 4. GROUP BY
        gen_grp = (gen_tree.find(exp.Group).sql(dialect="sqlite").lower()) if gen_tree.find(exp.Group) else ""
        gold_grp = (gold_tree.find(exp.Group).sql(dialect="sqlite").lower()) if gold_tree.find(exp.Group) else ""
        results["group_by"] = (gen_grp == gold_grp)

        # 5. HAVING
        gen_hav = (gen_tree.find(exp.Having).sql(dialect="sqlite").lower()) if gen_tree.find(exp.Having) else ""
        gold_hav = (gold_tree.find(exp.Having).sql(dialect="sqlite").lower()) if gold_tree.find(exp.Having) else ""
        results["having"] = (gen_hav == gold_hav)

        # 6. ORDER BY
        gen_ord = (gen_tree.find(exp.Order).sql(dialect="sqlite").lower()) if gen_tree.find(exp.Order) else ""
        gold_ord = (gold_tree.find(exp.Order).sql(dialect="sqlite").lower()) if gold_tree.find(exp.Order) else ""
        results["order_by"] = (gen_ord == gold_ord)

        # 7. LIMIT
        gen_lim = (gen_tree.find(exp.Limit).sql(dialect="sqlite").lower()) if gen_tree.find(exp.Limit) else ""
        gold_lim = (gold_tree.find(exp.Limit).sql(dialect="sqlite").lower()) if gold_tree.find(exp.Limit) else ""
        results["limit"] = (gen_lim == gold_lim)

        return results

    def execute_safely(self, sql_str: str) -> Tuple[bool, Any, float]:
        """Executes a SQL query in read-only SQLite mode with latency timing."""
        start = time.perf_counter()
        try:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            cursor = conn.cursor()
            cursor.execute(sql_str)
            rows = cursor.fetchall()
            conn.close()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return True, rows, elapsed_ms
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return False, str(e), elapsed_ms

    @staticmethod
    def compare_row_sets(
        gen_rows: List[Tuple],
        gold_rows: List[Tuple],
        ordered: bool = False,
        tolerance: float = 1e-4,
    ) -> bool:
        """
        Compares query output tuples with float rounding tolerance.
        If ordered is False, compares sorted lists to remain order-agnostic.
        """
        if len(gen_rows) != len(gold_rows):
            return False

        def normalize_cell(val):
            if isinstance(val, float):
                return round(val, 2)
            return val

        def normalize_row(row):
            return tuple(normalize_cell(c) for c in row)

        norm_gen = [normalize_row(r) for r in gen_rows]
        norm_gold = [normalize_row(r) for r in gold_rows]

        if not ordered:
            try:
                norm_gen = sorted(norm_gen)
                norm_gold = sorted(norm_gold)
            except TypeError:
                pass  # Fallback to direct comparison if types cannot be compared

        return norm_gen == norm_gold

    def evaluate_query(
        self,
        test_case: SQLTestCase,
        generated_sql_or_response: str,
    ) -> SQLTestResult:
        """Evaluates a generated SQL string or model response against the gold test case."""
        raw_gen_sql = self.extract_sql(generated_sql_or_response)
        raw_gold_sql = self.extract_sql(test_case.gold_sql)

        # 1. AST Canonicalization
        gen_ast, gen_err = self.canonicalize_ast(raw_gen_sql)
        gold_ast, _ = self.canonicalize_ast(raw_gold_sql)

        canon_gen_str = gen_ast.sql(dialect="sqlite").lower() if gen_ast else ""
        canon_gold_str = gold_ast.sql(dialect="sqlite").lower() if gold_ast else ""

        # 2. Safety Verification
        if gen_ast:
            is_safe, safety_err = self.verify_safety(gen_ast)
            if not is_safe:
                return SQLTestResult(
                    test_id=test_case.id,
                    category=test_case.category,
                    prompt=test_case.prompt,
                    generated_sql=raw_gen_sql,
                    gold_sql=raw_gold_sql,
                    canonical_gen_sql=canon_gen_str,
                    canonical_gold_sql=canon_gold_str,
                    exact_ast_match=False,
                    component_matches={},
                    exec_accuracy=False,
                    exec_success=False,
                    latency_ms=0.0,
                    error=safety_err,
                )

        # 3. Exact AST Match & Component Matching
        exact_ast_match = (canon_gen_str != "") and (canon_gen_str == canon_gold_str)
        comp_matches = self.compare_components(gen_ast, gold_ast)

        # 4. Safe Execution Verification
        gold_ok, gold_rows, _ = self.execute_safely(raw_gold_sql)
        gen_ok, gen_rows_or_err, latency = self.execute_safely(raw_gen_sql)

        exec_accuracy = False
        error_msg = None

        if not gen_ok:
            error_msg = str(gen_rows_or_err)
        elif gold_ok:
            exec_accuracy = self.compare_row_sets(
                gen_rows_or_err,
                gold_rows,
                ordered=test_case.ordered,
            )

        return SQLTestResult(
            test_id=test_case.id,
            category=test_case.category,
            prompt=test_case.prompt,
            generated_sql=raw_gen_sql,
            gold_sql=raw_gold_sql,
            canonical_gen_sql=canon_gen_str,
            canonical_gold_sql=canon_gold_str,
            exact_ast_match=exact_ast_match,
            component_matches=comp_matches,
            exec_accuracy=exec_accuracy,
            exec_success=gen_ok,
            latency_ms=latency,
            error=error_msg,
            gen_rows=gen_rows_or_err if gen_ok else None,
            gold_rows=gold_rows if gold_ok else None,
        )

    def print_summary(self, results: List[SQLTestResult]) -> None:
        """Renders a comprehensive diagnostic evaluation report to the terminal using Rich."""
        table = Table(title="🔍 SQL Exact Match & Execution Evaluation Results", show_lines=True)
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Category", style="magenta")
        table.add_column("AST Match", justify="center")
        table.add_column("Exec Acc", justify="center")
        table.add_column("Comp Matches", style="dim")
        table.add_column("Latency", justify="right")
        table.add_column("Status / Error", style="yellow")

        ast_pass = sum(1 for r in results if r.exact_ast_match)
        exec_pass = sum(1 for r in results if r.exec_accuracy)
        total = len(results)

        for r in results:
            ast_icon = "[green]PASS[/green]" if r.exact_ast_match else "[red]FAIL[/red]"
            exec_icon = "[bold green]PASS[/bold green]" if r.exec_accuracy else "[bold red]FAIL[/bold red]"

            # Summarize components
            passed_comps = [k.upper() for k, v in r.component_matches.items() if v]
            comp_summary = ",".join(passed_comps) if passed_comps else "None"

            status = "[green]OK[/green]" if r.exec_success else f"[red]{r.error}[/red]"
            table.add_row(
                r.test_id,
                r.category,
                ast_icon,
                exec_icon,
                comp_summary,
                f"{r.latency_ms:.1f}ms",
                status,
            )

        self.console.print(table)
        self.console.print(
            f"\n[bold]Summary:[/bold] "
            f"Exact AST Match (EM): [bold green]{ast_pass}/{total} ({ast_pass/total*100:.1f}%)[/bold green] | "
            f"Execution Accuracy (EX): [bold green]{exec_pass}/{total} ({exec_pass/total*100:.1f}%)[/bold green]"
        )


    def run_live_benchmark(
        self,
        query_engine=None,
        suite: Optional[List[SQLTestCase]] = None,
        save_report_path: Optional[Union[Path, str]] = None,
    ) -> List[SQLTestResult]:
        """Runs the benchmark suite against a live NLSQL query engine and collects tracking results."""
        if query_engine is None:
            from src.engines.model_factory import initialize_models
            from src.engines.structured_engine import StructuredEngineManager

            llm, _ = initialize_models()
            mgr = StructuredEngineManager(db_path=self.db_path)
            query_engine = mgr.get_query_engine(llm=llm)

        cases = suite or SQL_BENCHMARK_SUITE
        self.console.print(f"[bold cyan]🚀 Starting Live Text-to-SQL Benchmark ({len(cases)} questions)...[/bold cyan]")

        results: List[SQLTestResult] = []
        for i, tc in enumerate(cases, 1):
            self.console.print(f"  [{i}/{len(cases)}] [yellow]Evaluating {tc.id}:[/yellow] {tc.prompt[:60]}...")
            try:
                resp = query_engine.query(tc.prompt)
                gen_sql = resp.metadata.get("sql_query") if hasattr(resp, "metadata") and resp.metadata else None
                if not gen_sql:
                    gen_sql = str(resp.response) if hasattr(resp, "response") else str(resp)

                res = self.evaluate_query(tc, gen_sql)
            except Exception as e:
                res = SQLTestResult(
                    test_id=tc.id,
                    category=tc.category,
                    prompt=tc.prompt,
                    generated_sql="",
                    gold_sql=tc.gold_sql,
                    canonical_gen_sql="",
                    canonical_gold_sql="",
                    exact_ast_match=False,
                    component_matches={},
                    exec_accuracy=False,
                    exec_success=False,
                    latency_ms=0.0,
                    error=str(e),
                )
            results.append(res)

        self.print_summary(results)

        if save_report_path:
            self.export_report(results, Path(save_report_path))

        return results

    def export_report(self, results: List[SQLTestResult], out_path: Path) -> None:
        """Exports a detailed markdown and json audit report for benchmark tracking."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        total = len(results)
        ast_pass = sum(1 for r in results if r.exact_ast_match)
        exec_pass = sum(1 for r in results if r.exec_accuracy)
        success_pass = sum(1 for r in results if r.exec_success)

        md_lines = [
            "# Text-to-SQL Benchmark Evaluation Report",
            f"\n**Total Cases**: {total} | **Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"- **Exact AST Match (EM)**: {ast_pass}/{total} ({ast_pass/total*100:.1f}%)",
            f"- **Execution Accuracy (EX)**: {exec_pass}/{total} ({exec_pass/total*100:.1f}%)",
            f"- **Execution Success Rate**: {success_pass}/{total} ({success_pass/total*100:.1f}%)\n",
            "| ID | Category | AST Match | Exec Accuracy | Latency | Generated SQL | Error |",
            "| :--- | :--- | :---: | :---: | :---: | :--- | :--- |",
        ]

        for r in results:
            ast_str = "✅ PASS" if r.exact_ast_match else "❌ FAIL"
            ex_str = "✅ PASS" if r.exec_accuracy else "❌ FAIL"
            clean_gen = r.generated_sql.replace("\n", " ").replace("|", "\\|")
            err = (r.error or "").replace("\n", " ")
            md_lines.append(
                f"| `{r.test_id}` | {r.category} | {ast_str} | {ex_str} | {r.latency_ms:.1f}ms | `{clean_gen[:60]}...` | {err} |"
            )

        out_path.write_text("\n".join(md_lines), encoding="utf-8")
        self.console.print(f"[bold green]📄 Benchmark report saved to: {out_path}[/bold green]")


if __name__ == "__main__":
    import argparse

    configure_clean_logging()
    parser = argparse.ArgumentParser(description="SQL Benchmark Evaluator")
    parser.add_argument("--live", action="store_true", help="Run against live LLM Text-to-SQL engine")
    parser.add_argument("--report", type=str, default="storage/sql_benchmark_results.md", help="Report save path")
    args = parser.parse_args()

    evaluator = SQLASTEvaluator()
    if args.live:
        evaluator.run_live_benchmark(save_report_path=args.report)
    else:
        print("Running sanity self-validation on 20 gold test queries...")
        self_results = [evaluator.evaluate_query(tc, tc.gold_sql) for tc in SQL_BENCHMARK_SUITE]
        evaluator.print_summary(self_results)
