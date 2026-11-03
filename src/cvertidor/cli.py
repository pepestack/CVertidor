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

    graph = subparsers.add_parser("graph", help="Exporta el diagrama del grafo de LangGraph en Mermaid")
    graph.add_argument("--salida", help="Archivo .mmd o .md donde guardarlo (por defecto se imprime)")
    return parser


def cmd_graph(args: argparse.Namespace) -> int:
    from cvertidor.graph import graph_mermaid

    mermaid = graph_mermaid()
    if not args.salida:
        print(mermaid)
        return 0
    path = Path(args.salida)
    path.parent.mkdir(parents=True, exist_ok=True)
    # En .md se envuelve en un bloque ```mermaid para que GitHub lo dibuje
    content = f"```mermaid\n{mermaid}\n```\n" if path.suffix == ".md" else mermaid
    path.write_text(content, encoding="utf-8")
    print(f"Diagrama guardado en {path}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    import psycopg

    from cvertidor.config import get_settings
    from cvertidor.ingestion.loaders import DocumentLoadError, load_text
    from cvertidor.llm.check import error_status
    from cvertidor.llm.providers import ModelConfigError
    from cvertidor.graph.runner import run_agent
    from cvertidor.pipeline import GenerationError, InputError

    settings = get_settings()
    if args.salida:
        settings = settings.model_copy(update={"output_dir": Path(args.salida)})

    try:
        cv_text, job_text = load_text(args.cv), load_text(args.job)
        result = run_agent(cv_text, job_text, settings, on_step=lambda m: print(f"-> {m}"))
    except (DocumentLoadError, InputError, ModelConfigError, GenerationError) as error:
        print(f"Error: {error}")
        return 2
    except psycopg.OperationalError as error:
        print("Error: no se pudo conectar a la base de datos (DB_URL). "
              "Si es local, levántela con: docker compose up -d")
        print(f"Detalle: {str(error).strip()[:200]}")
        return 2
    except Exception as error:
        if error_status(error) != 429:
            raise
        # Todos los modelos de la cadena respondieron 429
        message = str(error)
        if "per-day" in message:
            print("Error: se agotó el cupo diario de modelos gratuitos de la cuenta de OpenRouter. "
                  "Se restablece cada día; con créditos en la cuenta el cupo diario es mayor.")
        elif "RESOURCE_EXHAUSTED" in message:
            print("Error: se agotó la cuota de la API de Gemini (RESOURCE_EXHAUSTED). "
                  "Revise los límites de su clave en Google AI Studio o reintente más tarde.")
        else:
            print("Error: los modelos configurados están saturados (429). Reintente en unos minutos.")
        print(f"Detalle: {str(error).strip()[:200]}")
        return 3

    plan = result.plan
    print(f"\nCandidato: {result.parsed_cv.candidate_name} ({result.candidate_id})")
    print(f"Oferta: {result.job_reqs.title or 'sin título'} | "
          f"{len(result.job_reqs.must_have)} requisitos obligatorios, {len(result.job_reqs.nice_to_have)} deseables")
    coverage = f"{plan.skill_coverage:.0%}" if plan.skill_coverage is not None else "sin requisitos"
    print(f"Plan: salida en {plan.output_language}{' (traducción)' if plan.translate else ''}, "
          f"cobertura de requisitos {coverage}, prioridad {' > '.join(plan.section_priority)}")
    if plan.missing_requirements:
        print(f"  Requisitos ausentes en el CV: {', '.join(plan.missing_requirements)}")
    print(f"Fragmentos: {result.sync.added} nuevos, {result.sync.removed} eliminados, "
          f"{result.sync.unchanged} sin cambios (embeddings: {result.embeddings_model})")
    print(f"Recuperados para la oferta: {len(result.retrieved_docs)} "
          f"en {result.retrieval_attempts} intento{'s' if result.retrieval_attempts != 1 else ''}")
    for m in result.metrics:
        tokens = f", {m.input_tokens} -> {m.output_tokens} tokens" if m.input_tokens is not None else ""
        model = f", {m.model}" if m.model else ""
        print(f"  {m.step:<9} {m.seconds:5.1f} s{model}{tokens}")
    for warning in result.warnings:
        print(f"Aviso: {warning}")
    print("\nArchivos generados:")
    for path in result.output_paths:
        print(f"  {path}")
    print(f"Auditoría: {settings.log_dir / 'audit.jsonl'} (run_id {result.run_id})")
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
    "graph": cmd_graph,
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
