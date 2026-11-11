import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.evaluation.sql_eval import SQLASTEvaluator, SQLTestCase, SQL_BENCHMARK_SUITE


@pytest.fixture
def evaluator():
    return SQLASTEvaluator()


def test_suite_coverage(evaluator):
    """Ensure benchmark suite contains all 20 calibrated cases."""
    assert len(SQL_BENCHMARK_SUITE) == 20
    categories = {tc.category for tc in SQL_BENCHMARK_SUITE}
    assert categories == {"Simple", "Joins", "Aggregations", "Complex"}


def test_gold_sql_execution(evaluator):
    """All 20 gold SQL queries must execute successfully and match expected row counts."""
    for tc in SQL_BENCHMARK_SUITE:
        ok, rows, _ = evaluator.execute_safely(tc.gold_sql)
        assert ok is True, f"Gold query for {tc.id} failed: {rows}"
        assert len(rows) == tc.expected_row_count, (
            f"Expected {tc.expected_row_count} rows for {tc.id}, got {len(rows)}"
        )


def test_ast_alias_normalization(evaluator):
    """Tests that varying table aliases (o/c vs ord/cust) produce identical canonical AST."""
    gold = "SELECT o.order_id, o.total_amount FROM orders o JOIN customers c ON o.customer_id = c.customer_id WHERE c.name = 'Alice Chen';"
    candidate = "SELECT ord.order_id, ord.total_amount FROM orders AS ord JOIN customers AS cust ON ord.customer_id = cust.customer_id WHERE cust.name = 'Alice Chen'"

    tc = SQLTestCase(
        id="TEST-ALIAS",
        prompt="dummy",
        gold_sql=gold,
        category="Joins",
        expected_row_count=2,
        description="alias test",
    )
    result = evaluator.evaluate_query(tc, candidate)
    assert result.exact_ast_match is True
    assert result.exec_accuracy is True


def test_ast_commutative_where_order(evaluator):
    """Tests that reordering AND conditions in WHERE produces identical canonical AST."""
    gold = "SELECT name, price FROM products WHERE category = 'Audio' AND status = 'Active';"
    candidate = "select name, price from products where status = 'Active' and category = 'Audio'"

    tc = SQLTestCase(
        id="TEST-COMM",
        prompt="dummy",
        gold_sql=gold,
        category="Simple",
        expected_row_count=2,
        description="commutative test",
    )
    result = evaluator.evaluate_query(tc, candidate)
    assert result.exact_ast_match is True
    assert result.exec_accuracy is True


def test_ast_safety_detection(evaluator):
    """Dangerous / Mutating queries must be blocked by the AST layer."""
    tc = SQLTestCase(
        id="TEST-SAFETY",
        prompt="dummy",
        gold_sql="SELECT name FROM products;",
        category="Simple",
        expected_row_count=6,
        description="safety test",
    )
    result = evaluator.evaluate_query(tc, "DROP TABLE products;")
    assert result.exact_ast_match is False
    assert result.exec_success is False
    assert "Prohibited SQL statement detected" in (result.error or "")


def test_component_breakdown(evaluator):
    """Test that individual clause mismatches are isolated correctly."""
    gold = "SELECT name, price FROM products WHERE price > 50;"
    candidate = "SELECT name, price FROM products WHERE price > 100;"

    tc = SQLTestCase(
        id="TEST-COMP",
        prompt="dummy",
        gold_sql=gold,
        category="Simple",
        expected_row_count=5,
        description="component test",
    )
    result = evaluator.evaluate_query(tc, candidate)
    assert result.exact_ast_match is False
    assert result.component_matches["select"] is True
    assert result.component_matches["from"] is True
    assert result.component_matches["where"] is False
