"""Nodos del grafo. Cada uno recibe el estado y el contexto de ejecución y devuelve la
actualización del estado; la clave opcional "audit" va al registro de auditoría y no al estado."""

import time
from typing import Any

from cvertidor.graph.edges import MAX_RETRIEVAL_ATTEMPTS, has_work_history
from cvertidor.graph.planning import build_plan
from cvertidor.graph.state import AgentContext, AgentState
from cvertidor.ingestion.pii import mask_pii
from cvertidor.llm.providers import get_structured_model
from cvertidor.observability import anonymize
from cvertidor.pipeline import (
    GenerationError,
    generate_adapted_cv,
    invoke_structured,
    parse_cv,
    save_outputs,
    step_metrics,
)
from cvertidor.prompts.job_analysis import JOB_ANALYSIS_PROMPT
from cvertidor.rag.chunking import create_cv_documents, make_candidate_id
from cvertidor.rag.retrieval import ALWAYS_INCLUDED_SECTIONS, build_retriever, format_docs, similarity_threshold
from cvertidor.rag.vector_store import sync_candidate_docs
from cvertidor.render import cv_to_html, cv_to_markdown
from cvertidor.schemas.job import JobRequirements

# Cada reintento de recuperación baja el umbral de similitud en este valor, hasta el mínimo.
# Es pequeño porque la escala de similitud de los embeddings de Gemini es estrecha
RETRIEVAL_THRESHOLD_STEP = 0.05
MIN_RETRIEVAL_THRESHOLD = 0.1


def _model_name(raw: Any) -> str | None:
    return (getattr(raw, "response_metadata", None) or {}).get("model_name")


def ingest_cv(state: AgentState, ctx: AgentContext) -> dict:
    """Estructura el CV (con la PII enmascarada) e indexa sus fragmentos en pgvector"""
    start = time.perf_counter()
    parsed_cv, removed, raw = parse_cv(state["cv_text"], ctx.settings)
    metrics = [step_metrics("parse", start, raw)]
    warnings = []
    if removed:
        warnings.append("Se descartaron datos de contacto que no están en el CV original: " + ", ".join(removed))

    candidate_id = make_candidate_id(parsed_cv.candidate_name)
    docs = create_cv_documents(parsed_cv)

    start = time.perf_counter()
    info = ctx.embeddings()
    if info.is_fallback:
        warnings.append(f"Se usaron embeddings locales ({info.model}) porque el proveedor no respondió.")
    sync = sync_candidate_docs(ctx.vector_store(), ctx.settings.require_db_url(), candidate_id, docs)
    metrics.append(step_metrics("index", start))

    return {
        "candidate_id": candidate_id,
        "parsed_cv": parsed_cv,
        "sync": sync,
        "warnings": warnings,
        "metrics": metrics,
        "audit": {
            "candidate": anonymize(candidate_id),
            "model": _model_name(raw),
            "chunks": len(docs),
            "added": sync.added,
            "removed": sync.removed,
            "unchanged": sync.unchanged,
            "contact_discarded": len(removed),
            "embeddings": info.model,
        },
    }


def analyze_job(state: AgentState, ctx: AgentContext) -> dict:
    """Extrae los requisitos de la oferta. Si el modelo no entrega un resultado válido el
    flujo sigue sin requisitos: el plan usa entonces solo la oferta completa"""
    start = time.perf_counter()
    chain = JOB_ANALYSIS_PROMPT | get_structured_model("job", JobRequirements, include_raw=True, settings=ctx.settings)
    try:
        job_reqs, raw = invoke_structured(chain, {"job_description": mask_pii(state["job_text"]).text}, "JobRequirements")
        decision, warnings = "ok", []
    except GenerationError as error:
        job_reqs, raw = JobRequirements(), None
        decision = "sin_requisitos"
        warnings = [f"No se pudieron extraer los requisitos de la oferta; se usará la oferta completa ({error})."]

    return {
        "job_reqs": job_reqs,
        "warnings": warnings,
        "metrics": [step_metrics("job", start, raw)],
        "audit": {
            "decision": decision,
            "model": _model_name(raw),
            "must_have": len(job_reqs.must_have),
            "nice_to_have": len(job_reqs.nice_to_have),
            "language": job_reqs.language,
        },
    }


def plan(state: AgentState, ctx: AgentContext) -> dict:
    """Decide el idioma de salida, el orden de las secciones y las consultas de recuperación"""
    result = build_plan(state["parsed_cv"], state["job_reqs"], state["job_text"], state["cv_text"])
    return {
        "plan": result,
        "retrieval_attempts": 0,
        "retrieved_docs": [],
        "audit": {
            "output_language": result.output_language,
            "translate": result.translate,
            "skill_coverage": result.skill_coverage,
            "missing": len(result.missing_requirements),
            "section_priority": result.section_priority,
            "retrieval_queries": len(result.retrieval_queries),
        },
    }


def retrieve(state: AgentState, ctx: AgentContext) -> dict:
    """Recupera los fragmentos relevantes. En cada reintento usa la siguiente consulta del
    plan y un umbral de similitud más bajo; conserva lo recuperado en intentos anteriores"""
    start = time.perf_counter()
    attempt = state.get("retrieval_attempts", 0)
    queries = state["plan"].retrieval_queries or [state["job_text"]]
    query = queries[min(attempt, len(queries) - 1)]

    info = ctx.embeddings()
    base = similarity_threshold(info.slug)
    threshold = max(MIN_RETRIEVAL_THRESHOLD, round(base - attempt * RETRIEVAL_THRESHOLD_STEP, 2))
    retriever = build_retriever(ctx.vector_store(), info.embeddings, state["candidate_id"], threshold)
    found = retriever.invoke(query)

    seen = set()
    docs = []
    for doc in [*state.get("retrieved_docs", []), *found]:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            docs.append(doc)

    attempts = attempt + 1
    enough = has_work_history(docs)
    warnings = []
    if not enough and attempts >= MAX_RETRIEVAL_ATTEMPTS:
        decision = "generar_sin_experiencia"
        warnings.append("Ninguna experiencia laboral superó el filtro de similitud, ni con consultas más amplias.")
    else:
        decision = "suficiente" if enough else "reintentar"

    return {
        "retrieved_docs": docs,
        "retrieval_attempts": attempts,
        "warnings": warnings,
        "metrics": [step_metrics(f"retrieve{attempts}" if attempts > 1 else "retrieve", start)],
        "audit": {
            "attempt": attempts,
            "query_index": min(attempt, len(queries) - 1),
            "threshold": threshold,
            "found": len(found),
            "total": len(docs),
            "work_history": sum(d.metadata.get("section") == "work_history" for d in docs),
            "decision": decision,
        },
    }


def generate(state: AgentState, ctx: AgentContext) -> dict:
    start = time.perf_counter()
    always_included = [d for d in create_cv_documents(state["parsed_cv"])
                       if d.metadata["section"] in ALWAYS_INCLUDED_SECTIONS]
    context = format_docs(state.get("retrieved_docs", []) + always_included)
    draft, raw = generate_adapted_cv(context, state["job_text"], state["parsed_cv"], ctx.settings,
                                     plan=state["plan"].instructions)
    metrics = step_metrics("generate", start, raw)
    return {
        "draft_cv": draft,
        "metrics": [metrics],
        "audit": {
            "model": metrics.model,
            "input_tokens": metrics.input_tokens,
            "output_tokens": metrics.output_tokens,
            "roles": len(draft.work_history),
            "max_responsibilities": max((len(w.responsibilities) for w in draft.work_history), default=0),
        },
    }


def render_and_save(state: AgentState, ctx: AgentContext) -> dict:
    lang = state["plan"].output_language
    markdown = cv_to_markdown(state["draft_cv"], lang=lang)
    html = cv_to_html(state["draft_cv"], lang=lang)
    paths = save_outputs(state["candidate_id"], markdown, html, ctx.settings.output_dir)
    return {
        "markdown": markdown,
        "html": html,
        "output_paths": [str(p) for p in paths],
        # Los nombres de archivo llevan el nombre del candidato: solo se registran las extensiones
        "audit": {"files": [p.suffix for p in paths], "language": lang},
    }
