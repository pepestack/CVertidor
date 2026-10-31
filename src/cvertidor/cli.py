"""Interfaz de línea de comandos de CVertidor."""

import argparse
import logging
from pathlib import Path

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

    run = subparsers.add_parser("run", help="Genera un CV adaptado a una oferta de trabajo")
    run.add_argument("--cv", required=True, help="Ruta al CV (.txt, .md, .pdf o .docx)")
    run.add_argument("--job", required=True, help="Ruta a la oferta de trabajo (.txt, .md, .pdf o .docx)")
    run.add_argument("--salida", help="Carpeta de salida (por defecto OUTPUT_DIR o ./output)")
    return parser


def cmd_run(args: argparse.Namespace) -> int:
    import psycopg

    from cvertidor.config import get_settings
    from cvertidor.ingestion.loaders import DocumentLoadError, load_text
    from cvertidor.llm.providers import ModelConfigError
    from cvertidor.pipeline import GenerationError, InputError, run_pipeline

    settings = get_settings()
    if args.salida:
        settings = settings.model_copy(update={"output_dir": Path(args.salida)})

    try:
        cv_text, job_text = load_text(args.cv), load_text(args.job)
        result = run_pipeline(cv_text, job_text, settings, on_step=lambda m: print(f"-> {m}"))
    except (DocumentLoadError, InputError, ModelConfigError, GenerationError) as error:
        print(f"Error: {error}")
        return 2
    except psycopg.OperationalError as error:
        print("Error: no se pudo conectar a la base de datos (DB_URL). "
              "Si es local, levántela con: docker compose up -d")
        print(f"Detalle: {str(error).strip()[:200]}")
        return 2
    except Exception as error:
        if getattr(error, "status_code", None) != 429:
            raise
        # Todos los modelos de la cadena respondieron 429
        if "per-day" in str(error):
            print("Error: se agotó el cupo diario de modelos gratuitos de la cuenta de OpenRouter. "
                  "Se restablece cada día; con créditos en la cuenta el cupo diario es mayor.")
        else:
            print("Error: los modelos configurados están saturados (429). Reintente en unos minutos.")
        print(f"Detalle: {str(error).strip()[:200]}")
        return 3

    print(f"\nCandidato: {result.parsed_cv.candidate_name} ({result.candidate_id})")
    print(f"Fragmentos: {result.sync.added} nuevos, {result.sync.removed} eliminados, "
          f"{result.sync.unchanged} sin cambios (embeddings: {result.embeddings_model})")
    print(f"Recuperados para la oferta: {len(result.retrieved_docs)}")
    for m in result.metrics:
        tokens = f", {m.input_tokens} -> {m.output_tokens} tokens" if m.input_tokens is not None else ""
        model = f", {m.model}" if m.model else ""
        print(f"  {m.step:<9} {m.seconds:5.1f} s{model}{tokens}")
    for warning in result.warnings:
        print(f"Aviso: {warning}")
    print("\nArchivos generados:")
    for path in result.output_paths:
        print(f"  {path}")
    return 0


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
    "run": cmd_run,
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
