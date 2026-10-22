import time
from dataclasses import dataclass
from typing import List, Dict, Any
from rich.console import Console
from rich.table import Table
from src.agent.document_agent import SpecializedDocumentAgent


@dataclass
class EvalTestCase:
    id: str
    category: str  # "Structured SQL", "Unstructured RAG", "Multi-Hop Synthesis"
    query: str
    expected_keywords: List[str]
    description: str


EVAL_TEST_CASES = [
    EvalTestCase(
        id="TC-01",
        category="Structured SQL",
        query="How many units of AeroSound Pro Wireless ANC Headphones (SKU-AUD-001) are in stock, and what is its retail price?",
        expected_keywords=["12", "199.99"],
        description="Verify Text-to-SQL retrieves exact stock count and unit price from products table.",
    ),
    EvalTestCase(
        id="TC-02",
        category="Unstructured RAG",
        query="What is the return window for a VIP tier customer compared to a Standard tier customer according to the return policy?",
        expected_keywords=["60", "30", "VIP"],
        description="Verify semantic retrieval extracts tier-specific return windows from POL-RET-2026-V3.",
    ),
    EvalTestCase(
        id="TC-03",
        category="Multi-Hop Synthesis",
        query="Customer Alice Chen (Customer ID 101) wants to return order #5001. She received it on 2026-09-10. Is she eligible for a full refund today?",
        expected_keywords=["VIP", "eligible", "60", "POL-RET-2026-V3"],
        description="Joint multi-hop reasoning: checks Alice's VIP tier in SQL, correlates with 60-day window in policy, and calculates days.",
    ),
    EvalTestCase(
        id="TC-04",
        category="Policy Exception",
        query="Can an opened box of AeroBuds True Wireless In-Ear Earphones be returned after 20 days if the customer simply changed their mind?",
        expected_keywords=["14", "hygiene", "unopened"],
        description="Verify sanitary/hygiene exception policy for opened in-ear audio equipment.",
    ),
    EvalTestCase(
        id="TC-05",
        category="Specs & Reviews Synthesis",
        query="What are customers complaining about regarding the AeroSound Pro headphones, and what is the troubleshooting step to reset Bluetooth pairing?",
        expected_keywords=["cushion", "8 seconds", "power"],
        description="Cross-document synthesis: extracts customer sentiment from reviews and reset steps from specs.",
    ),
]


def run_agent_benchmark(agent: SpecializedDocumentAgent = None) -> List[Dict[str, Any]]:
    """
    Runs the comprehensive evaluation benchmark across all test cases.
    Evaluates keyword presence, source grounding, and execution latency.
    """
    console = Console()
    if agent is None:
        agent = SpecializedDocumentAgent()

    console.print("\n[bold cyan]🚀 Starting Nexus Specialized Document Agent Benchmark...[/bold cyan]\n")

    results = []
    table = Table(title="Benchmark Results: Nexus Document Agent")
    table.add_column("Test ID", style="bold yellow")
    table.add_column("Category", style="cyan")
    table.add_column("Latency", justify="right")
    table.add_column("Keywords Hit", justify="center")
    table.add_column("Status", style="bold")

    for case in EVAL_TEST_CASES:
        console.print(f"[dim]Executing {case.id}: {case.query[:60]}...[/dim]")
        start = time.time()
        response = agent.chat(case.query)
        latency = round(time.time() - start, 2)

        resp_lower = response.response_text.lower()
        matched_keywords = [kw for kw in case.expected_keywords if kw.lower() in resp_lower]
        match_rate = len(matched_keywords) / len(case.expected_keywords)

        passed = match_rate >= 0.5  # At least 50% critical semantic terms present
        status_text = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"

        table.add_row(
            case.id,
            case.category,
            f"{latency}s",
            f"{len(matched_keywords)}/{len(case.expected_keywords)}",
            status_text,
        )

        results.append({
            "id": case.id,
            "category": case.category,
            "query": case.query,
            "response": response.response_text,
            "sources": response.sources,
            "latency": latency,
            "matched_keywords": matched_keywords,
            "passed": passed,
        })

    console.print(table)
    total_passed = sum(1 for r in results if r["passed"])
    console.print(f"\n[bold]Total Passed: {total_passed}/{len(results)} ({round(total_passed/len(results)*100, 1)}%)[/bold]\n")

    return results


if __name__ == "__main__":
    run_agent_benchmark()
