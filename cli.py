import sys
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.table import Table

from src.agent.document_agent import SpecializedDocumentAgent
from src.evaluation.eval_suite import run_agent_benchmark
from src.config import config


def print_banner(console: Console):
    banner_text = """
[bold cyan]╔═════════════════════════════════════════════════════════════════════════╗[/bold cyan]
[bold cyan]║[/bold cyan]   [bold white]NEXUS SPECIALIZED DOCUMENT & DATA AGENT[/bold white]                             [bold cyan]║[/bold cyan]
[bold cyan]║[/bold cyan]   [dim]LlamaIndex Multi-Hop RAG over Structured (SQL) & Unstructured Vault[/dim]     [bold cyan]║[/bold cyan]
[bold cyan]╚═════════════════════════════════════════════════════════════════════════╝[/bold cyan]
    """
    console.print(banner_text)
    console.print("[dim]Type your query, or use commands: /vault, /eval, /schema, /clear, /exit[/dim]\n")


def display_vault_summary(console: Console, agent: SpecializedDocumentAgent):
    summary = agent.get_vault_summary()

    table = Table(title="Knowledge Vault Inventory", header_style="bold magenta")
    table.add_column("Vault Type", style="cyan")
    table.add_column("Resource / Table", style="white")
    table.add_column("Metrics / Details", style="green")

    for tbl, count in summary["structured_tables"].items():
        table.add_row("Structured (SQLite)", f"Table: {tbl}", f"{count} records")

    for doc in summary["unstructured_documents"]:
        table.add_row("Unstructured Vault", f"Doc: {doc}", "Indexed & Vectorized")

    console.print(table)


def main():
    console = Console()
    print_banner(console)

    with console.status("[bold green]Initializing LlamaIndex Document Agent and Vaults...[/bold green]"):
        agent = SpecializedDocumentAgent(verbose=False)

    while True:
        try:
            user_input = Prompt.ask("[bold green]Query[/bold green]").strip()

            if not user_input:
                continue

            if user_input.lower() in ["/exit", "exit", "quit", ":q"]:
                console.print("[yellow]Exiting Nexus Agent. Goodbye![/yellow]")
                sys.exit(0)

            elif user_input.lower() == "/clear":
                agent.reset_conversation()
                console.clear()
                print_banner(console)
                console.print("[green]Conversation memory cleared.[/green]\n")
                continue

            elif user_input.lower() == "/vault":
                display_vault_summary(console, agent)
                continue

            elif user_input.lower() == "/schema":
                schema_path = config.structured_dir / "schema.sql"
                if schema_path.exists():
                    with open(schema_path, "r", encoding="utf-8") as f:
                        console.print(Panel(f.read(), title="SQLite Schema DDL", border_style="cyan"))
                continue

            elif user_input.lower() == "/eval":
                run_agent_benchmark(agent)
                continue

            # Execute Agent Query
            with console.status("[bold cyan]Agent reasoning across SQL & Knowledge Vault...[/bold cyan]"):
                response = agent.chat(user_input)

            # Display Response
            console.print("\n" + "=" * 70)
            console.print(Panel(Markdown(response.response_text), title=f"[bold green]Nexus Agent Response[/bold green] (Latency: {response.latency_seconds}s)", border_style="green"))

            if response.sources:
                console.print(f"[bold cyan]Grounded Sources:[/bold cyan] {', '.join(response.sources)}")
            console.print("=" * 70 + "\n")

        except KeyboardInterrupt:
            console.print("\n[yellow]Session aborted by user. Exiting.[/yellow]")
            break
        except Exception as e:
            console.print(f"[bold red]Execution error:[/bold red] {e}")


if __name__ == "__main__":
    main()
