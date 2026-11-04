"""
Punto de entrada de CVertidor (consola).

Coordina todos los módulos y ejecuta el grafo cíclico:

    parse (Gemini) → chunkeo → ingesta pgvector → recuperación MMR
        → grafo LangGraph (generate → evaluate → [correct ⇄] → save)

Uso::

    uv run cvertidor --demo
    uv run cvertidor --cv cv.txt --job oferta.txt
    uv run python -m cvertidor --cv-text "Juan Pérez..." --job "Buscamos ..."
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

from cvertidor import database, graph
from cvertidor.config import MAX_ITERATIONS, validate
from cvertidor.llm import parse_raw_cv
from cvertidor.processor import create_cv_documents

# --- Materiales de ejemplo (del notebook original) ------------------------- #
SAMPLE_CV = """
Juan Pérez
Email: juan.perez@email.com | Tel: +56912345678 | Santiago, Chile
LinkedIn: linkedin.com/in/juanperez

Experiencia Laboral:
- Senior Data Engineer en TechCorp (2021 - Presente):
  Lideré la migración de pipelines ETL a Databricks, reduciendo tiempos de
  procesamiento en un 40%.
  Implementé arquitecturas de búsqueda vectorial y RAG utilizando
  PostgreSQL (pgvector) y LangChain.

Educación:
- Ingeniería Civil Informática, Universidad de Chile (2015 - 2020)

Habilidades Técnicas: Python, SQL, PostgreSQL, LangChain, Spark, Docker, GCP
Idiomas: Español (Nativo), Inglés (Avanzado)
Certificaciones: AWS Certified Solutions Architect (AWS) 2023
""".strip()

SAMPLE_JOB = (
    "Buscamos un Data Engineer con experiencia en Python, construcción de "
    "pipelines de datos y bases de datos vectoriales como pgvector."
)


def _read_input(value: str, flag: str, sample: str) -> str:
    """Resuelve una entrada: archivo (flag) o --demo con el material de ejemplo."""
    if value:
        return Path(value).read_text(encoding="utf-8")
    # En modo demo se usa el material incluido; si no hay ninguno, pide al usuario.
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cvertidor",
        description="Adapta CVs a ofertas de trabajo con RAG local y un grafo cíclico verificable.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_argument_group("Entradas")
    group.add_argument("--cv", metavar="ARCHIVO", help="Archivo de texto con el CV crudo.")
    group.add_argument("--cv-text", metavar="TEXTO", help="CV crudo inline (prioridad alta).")
    group.add_argument("--job", metavar="ARCHIVO", help="Archivo de texto con la oferta.")
    group.add_argument("--job-text", metavar="TEXTO", help="Oferta inline (prioridad alta).")
    group.add_argument("--demo", action="store_true", help="Usa el CV y la oferta de ejemplo.")
    group.add_argument(
        "--max-iterations",
        type=int,
        default=MAX_ITERATIONS,
        help="Tope de ciclos de corrección del grafo cíclico.",
    )

    args = parser.parse_args()

    # 1) Credenciales
    validate()

    # 2) Resolverse las entradas
    raw_cv = _read_input(args.cv_text or args.cv, "cv", SAMPLE_CV if args.demo else "")
    if not raw_cv:
        parser.error("Se requiere un CV: usa --cv, --cv-text o --demo.")
    job_description = _read_input(
        args.job_text or args.job, "job", SAMPLE_JOB if args.demo else ""
    )
    if not job_description:
        parser.error("Se requiere una oferta: usa --job, --job-text o --demo.")

    print("=== CVertidor: pipeline de adaptación de CV ===")
    print(f"Oferta: {job_description[:120]}{'...' if len(job_description) > 120 else ''}\n")

    # 3) Infraestructura vectorial
    print("[1/5] Configurando extensión vector en Supabase...")
    database.init_vector_extension()

    # 4) Parsing del CV crudo a StandardCV
    print("[2/5] Extrayendo información estructurada del CV (Gemini)...")
    parsed_cv = parse_raw_cv(raw_cv)
    print(
        f"   ✅ Extracción completada para: {parsed_cv.candidate_name} "
        f"(skills: {parsed_cv.technical_skills})"
    )

    # 5) Chunks + ingesta en pgvector
    print("[3/5] Generando chunk de documentos e ingestando en pgvector...")
    documents = create_cv_documents(parsed_cv)
    database.build_vector_store(documents, parsed_cv.candidate_name)
    print(f"   ✅ {len(documents)} chunks insertados para {parsed_cv.candidate_name}")

    # 6) Recuperación MMR del contexto relevante
    print("[4/5] Recuperando contexto relevante por MMR...")
    retriever = database.create_retriever(parsed_cv.candidate_name)
    retrieved_docs = retriever.invoke(job_description)
    context = graph.format_docs(retrieved_docs)
    for i, doc in enumerate(retrieved_docs, 1):
        preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
        print(f"   --- Chunk {i} [{doc.metadata.get('section', 'N/A')}] ---")
        print(textwrap.indent(preview, "    "))

    # 7) Ejecución del grafo cíclico
    print("[5/5] Ejecutando grafo LangGraph (generate ⇄ evaluate / correct → save)...")
    compiled = graph.build_graph()
    initial_state = {
        "job_description": job_description,
        "candidate_id": parsed_cv.candidate_name,
        "candidate_context": context,
        "max_iterations": args.max_iterations,
    }

    # Stream por "updates": cada iteración trae las salidas parciales de cada nodo
    # que se ejecutó en ese super-paso. Se acumulan para reconstruir el estado
    # final (con final_cv y save_result) sin re-ejecutar el grafo.
    trace: dict = {}
    for update in compiled.stream(initial_state, stream_mode="updates"):
        # `update` es un dict {nombre_nodo: estado_parcial}
        for node_name, node_state in update.items():
            status = node_state.get("status")
            if status:
                print(f"   • [{node_name}] {status}")
        for _, node_state in update.items():
            trace.update(node_state)

    final_cv = trace.get("final_cv")
    last_status = trace.get("status", "")

    # 8) Resultado final
    print(f"\n=== CV ADAPTADO FINAL ({last_status}) ===")
    print(f"Candidato: {parsed_cv.candidate_name}")
    print(f"Registro: {trace.get('save_result')}")
    print("\n=== JSON del CV ===")
    if final_cv is not None:
        print(final_cv.model_dump_json(indent=2))
    else:
        print("(No se obtuvo un CV final válido en el estado del grafo.)")


if __name__ == "__main__":
    main()
