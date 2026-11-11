"""Entry point package for llamaindex-document-agent CLI."""


def main() -> None:
    from cli import main as cli_main
    cli_main()
