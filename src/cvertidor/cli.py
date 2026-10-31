"""Interfaz de línea de comandos de CVertidor."""

import argparse

from cvertidor import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cvertidor",
        description="Adapta un currículum a una oferta de trabajo mediante RAG y LangGraph.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_subparsers(dest="command", metavar="comando")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
