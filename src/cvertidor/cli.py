"""Interfaz de línea de comandos de CVertidor."""

import argparse
import logging

from cvertidor import __version__
from cvertidor.config import ROLES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cvertidor",
        description="Adapta un currículum a una oferta de trabajo mediante RAG y LangGraph.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="comando")

    check = subparsers.add_parser(
        "check-models",
        help="Prueba cada modelo configurado (principal y fallbacks) con una llamada mínima",
    )
    check.add_argument("--rol", action="append", choices=ROLES, dest="roles",
                       help="Limita la prueba a un rol (se puede repetir)")
    check.add_argument("--sin-embeddings", action="store_true", help="No prueba el modelo de embeddings")
    return parser


def cmd_check_models(args: argparse.Namespace) -> int:
    from cvertidor.llm.check import check_embeddings, check_roles, format_report, roles_without_working_model
    from cvertidor.llm.providers import ModelConfigError

    print("Probando modelos configurados (una llamada mínima por modelo)...\n")
    try:
        results = check_roles(args.roles)
    except ModelConfigError as error:
        print(f"Error de configuración: {error}")
        return 2

    if not args.sin_embeddings:
        results.append(check_embeddings())
    print(format_report(results))

    broken = roles_without_working_model(results)
    print()
    if broken:
        print(f"Sin ningún modelo disponible para: {', '.join(broken)}. Cambie los modelos en .env.")
        return 1
    print("Todos los roles tienen al menos un modelo disponible.")
    return 0


COMMANDS = {
    "check-models": cmd_check_models,
}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
