"""Nodos del grafo. Cada uno recibe el estado y el contexto de ejecución y devuelve la
actualización del estado; la clave opcional "audit" va al registro de auditoría y no al estado."""

import time
from typing import Any

from langchain_core.documents import Document
from langgraph.config import get_store
from langgraph.types import interrupt

from cvertidor.graph.edges import MAX_RETRIEVAL_ATTEMPTS, has_work_history
from cvertidor.graph.planning import LANGUAGE_NAMES, build_plan
from cvertidor.graph.state import AgentContext, AgentState
from cvertidor.ingestion.pii import scrub_pii
from cvertidor.llm.check import describe_error
from cvertidor.llm.decisions import BEHAVIORS, DecisionError, evaluate_draft
from cvertidor.llm.providers import get_structured_model
from cvertidor.memory import store as memory
from cvertidor.observability import anonymize
from cvertidor.pipeline import (
    CANDIDATE_TOKEN,
    GenerationError,
    generate_adapted_cv,
    invoke_structured,
    mask_cv,
    parse_cv,
    restore_missing_sections,
    restore_missing_work_history,
    save_outputs,
    step_metrics,
)
from cvertidor.prompts.job_analysis import JOB_ANALYSIS_PROMPT
from cvertidor.prompts.judge import JUDGE_PROMPT
from cvertidor.rag.chunking import create_cv_documents, make_candidate_id
from cvertidor.rag.retrieval import ALWAYS_INCLUDED_SECTIONS, build_retriever, format_docs, similarity_threshold
from cvertidor.rag.vector_store import sync_candidate_docs
from cvertidor.render import cv_to_html, cv_to_markdown
from cvertidor.schemas.cv import StandardCV
from cvertidor.schemas.evaluation import JudgeReview, Verdict
from cvertidor.schemas.job import JobRequirements
from cvertidor.validation.checks import run_checks, status_from_scores

# Cada reintento de recuperación baja el umbral de similitud en este valor, hasta el mínimo.
# Es pequeño porque la escala de similitud de los embeddings de Gemini es estrecha
RETRIEVAL_THRESHOLD_STEP = 0.05
MIN_RETRIEVAL_THRESHOLD = 0.1

CONFIRMED_HEADING = "--- Correcciones confirmadas por el candidato ---"
REVIEW_OPTIONS = ["aprobar", "corregir", "descartar"]
ESCALATION_OPTIONS = ["aceptar", "corregir", "descartar"]
MAX_ESCALATION_RESETS = 1


def _model_name(raw: Any) -> str | None:
    return (getattr(raw, "response_metadata", None) or {}).get("model_name")


def load_memory(state: AgentState, ctx: AgentContext) -> dict:
    """Primera decisión del agente: reutiliza el CV y la oferta si ya los procesó antes"""
    store = get_store()
    cv_hash, job_hash = memory.text_hash(state["cv_text"]), memory.text_hash(state["job_text"])
    profile = None if ctx.reprocess else memory.load_profile(store, cv_hash)
    job_reqs = None if ctx.reprocess else memory.load_job(store, job_hash)

    update = {
        "runs_in_thread": state.get("runs_in_thread", 0) + 1,
        "cv_hash": cv_hash,
        "job_hash": job_hash,
        "cv_from_memory": profile is not None,
        "job_from_memory": job_reqs is not None,
    }
    if profile:
        update["candidate_id"], update["parsed_cv"] = profile
    if job_reqs:
        update["job_reqs"] = job_reqs
    update["audit"] = {
        "cv_from_memory": profile is not None,
        "job_from_memory": job_reqs is not None,
        "runs_in_thread": update["runs_in_thread"],
        "reprocess": ctx.reprocess,
    }
    return update


def ingest_cv(state: AgentState, ctx: AgentContext) -> dict:
    """Estructura el CV (con la PII enmascarada) y lo guarda en la memoria de largo plazo"""
    start = time.perf_counter()
    parsed_cv, removed, raw = parse_cv(state["cv_text"], ctx.settings)
    metrics = step_metrics("parse", start, raw)
    warnings = []
    if removed:
        warnings.append("Se descartaron datos de contacto que no están en el CV original: " + ", ".join(removed))

    candidate_id = make_candidate_id(parsed_cv.candidate_name)
    memory.save_profile(get_store(), candidate_id, state["cv_hash"], parsed_cv)
    return {
        "candidate_id": candidate_id,
        "parsed_cv": parsed_cv,
        "warnings": warnings,
        "metrics": [metrics],
        "audit": {
            "candidate": anonymize(candidate_id),
            "model": metrics.model,
            "input_tokens": metrics.input_tokens,
            "output_tokens": metrics.output_tokens,
            "contact_discarded": len(removed),
        },
    }


def index_cv(state: AgentState, ctx: AgentContext) -> dict:
    """Sincroniza los fragmentos en pgvector; sin cambios no se llama al modelo de embeddings"""
    start = time.perf_counter()
    candidate_id = state["candidate_id"]
    # Se indexa la copia enmascarada: los fragmentos van al modelo de embeddings y al de generación
    docs = create_cv_documents(mask_cv(state["parsed_cv"])[0], candidate_id)
    info = ctx.embeddings()
    warnings = []
    if info.is_fallback:
        warnings.append(f"Se usaron embeddings locales ({info.model}) porque el proveedor no respondió.")
    sync = sync_candidate_docs(ctx.vector_store(), ctx.settings.require_db_url(), candidate_id, docs)
    return {
        "sync": sync,
        "warnings": warnings,
        "metrics": [step_metrics("index", start)],
        "audit": {
            "candidate": anonymize(candidate_id),
            "chunks": len(docs),
            "added": sync.added,
            "removed": sync.removed,
            "unchanged": sync.unchanged,
            "embeddings": info.model,
        },
    }


def analyze_job(state: AgentState, ctx: AgentContext) -> dict:
    """Extrae los requisitos de la oferta. Si el modelo no entrega un resultado válido el
    flujo sigue sin requisitos: el plan usa entonces solo la oferta completa"""
    start = time.perf_counter()
    chain = JOB_ANALYSIS_PROMPT | get_structured_model("job", JobRequirements, include_raw=True, settings=ctx.settings)
    try:
        job_reqs, raw = invoke_structured(chain, {"job_description": scrub_pii(state["job_text"])}, "JobRequirements")
        decision, warnings = "ok", []
        memory.save_job(get_store(), state["job_hash"], job_reqs)
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
    """Decide el idioma de salida, el orden de las secciones y las consultas de recuperación,
    considerando lo que recuerda del candidato"""
    store, candidate_id = get_store(), state["candidate_id"]
    preferences = memory.load_preferences(store, candidate_id)
    warnings = []
    try:
        similar = memory.find_similar_generations(store, candidate_id, scrub_pii(state["job_text"]))
    except Exception as error:
        # El recuerdo semántico es opcional: si el modelo de embeddings falla, se sigue sin él
        similar = []
        warnings.append(f"No se pudieron consultar las generaciones anteriores ({type(error).__name__}).")

    result = build_plan(state["parsed_cv"], state["job_reqs"], state["job_text"], state["cv_text"], preferences,
                        memory.load_preferred_language(store, candidate_id))
    return {
        "plan": result,
        "preferences": preferences,
        "similar_generations": similar,
        "retrieval_attempts": 0,
        "retrieved_docs": [],
        "generation_attempts": 0,
        "draft_repeated": False,
        "generation_failed": False,
        "verdict": None,
        # El hilo conserva el estado: nada de una ejecución anterior debe parecer de esta
        "human_decision": None,
        "human_corrections": [],
        "escalation_resets": 0,
        "markdown": "",
        "html": "",
        "output_paths": [],
        "warnings": warnings,
        "audit": {
            "output_language": result.output_language,
            "translate": result.translate,
            "skill_coverage": result.skill_coverage,
            "missing": len(result.missing_requirements),
            "section_priority": result.section_priority,
            "retrieval_queries": len(result.retrieval_queries),
            "preferences": len(preferences),
            "similar_generations": [s["score"] for s in similar],
        },
    }


def retrieve(state: AgentState, ctx: AgentContext) -> dict:
    """Recupera los fragmentos relevantes. En cada reintento usa la siguiente consulta del
    plan y un umbral de similitud más bajo; conserva lo recuperado en intentos anteriores"""
    start = time.perf_counter()
    attempt = state.get("retrieval_attempts", 0)
    queries = state["plan"].retrieval_queries or [scrub_pii(state["job_text"])]
    query = queries[min(attempt, len(queries) - 1)]

    info = ctx.embeddings()
    base = similarity_threshold(info.slug)
    threshold = max(MIN_RETRIEVAL_THRESHOLD, round(base - attempt * RETRIEVAL_THRESHOLD_STEP, 2))
    retriever = build_retriever(ctx.vector_store(), info.embeddings, state["candidate_id"], threshold)
    # EmbeddingsFilter devuelve una subclase interna que el checkpointer no sabe serializar
    found = [Document(page_content=d.page_content, metadata=d.metadata) for d in retriever.invoke(query)]

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
    masked_cv = mask_cv(state["parsed_cv"])[0]
    always_included = [d for d in create_cv_documents(masked_cv, state["candidate_id"])
                       if d.metadata["section"] in ALWAYS_INCLUDED_SECTIONS]
    context = format_docs(state.get("retrieved_docs", []) + always_included)

    # Las correcciones van en el contexto: así las aceptan también el modelo de decisión y el juez
    corrections = state.get("human_corrections", [])
    if corrections:
        context += f"\n\n{CONFIRMED_HEADING}\n" + "\n".join(f"- {c}" for c in corrections)

    plan_text = state["plan"].instructions
    verdict = state.get("verdict")
    if verdict and verdict.feedback:
        plan_text += f"\n- Correcciones obligatorias del intento anterior:\n{verdict.feedback}"

    attempts = state.get("generation_attempts", 0) + 1
    # Con correcciones de una persona o de la validación se edita el borrador, en vez de reescribirlo.
    # `plan` y `edit_cv` reinician el veredicto: el borrador de una ejecución anterior no llega aquí
    current_draft = None
    if corrections or (verdict and verdict.feedback):
        current_draft = review_markdown(state["draft_cv"], state["plan"].output_language, state["parsed_cv"])
    try:
        draft, raw = generate_adapted_cv(context, state["job_text"], state["parsed_cv"], ctx.settings,
                                         plan=plan_text, current_draft=current_draft, human_edit=bool(corrections))
    except Exception as error:
        # Sin veredicto no hay un borrador de esta ejecución que conservar
        if verdict is None:
            raise
        return _keep_previous_draft(state, verdict, attempts, start, error)
    metrics =step_metrics("generate" if attempts == 1 else f"generate{attempts}", start, raw)
    warnings = []
    if not corrections and restore_missing_work_history(draft, state["parsed_cv"]):
        warnings.append("El modelo omitió toda la experiencia laboral: se agregaron los cargos, sin responsabilidades.")
    # Una persona puede haber pedido quitar una sección
    restored = [] if corrections else restore_missing_sections(draft, state["parsed_cv"], ALWAYS_INCLUDED_SECTIONS)
    if restored:
        names = ", ".join(RESTORED_SECTION_NAMES[name] for name in restored)
        untranslated = " (sin traducir)" if state["plan"].translate else ""
        warnings.append(f"El modelo omitió {names}: se copió del CV original{untranslated}.")
    repeated = attempts > 1 and draft == state.get("draft_cv")
    return {
        "context": context,
        "draft_cv": draft,
        "generation_attempts": attempts,
        "draft_repeated": repeated,
        "generation_failed": False,
        "warnings": warnings,
        "metrics": [metrics],
        "audit": {
            "attempt": attempts,
            "with_feedback": bool(verdict and verdict.feedback),
            "human_corrections": len(corrections),
            "restored_sections": restored,
            "repeated": repeated,
            "model": metrics.model,
            "input_tokens": metrics.input_tokens,
            "output_tokens": metrics.output_tokens,
            "roles": len(draft.work_history),
            "max_responsibilities": max((len(w.responsibilities) for w in draft.work_history), default=0),
        },
    }


def _keep_previous_draft(state: AgentState, verdict: Verdict, attempts: int, start: float, error: Exception) -> dict:
    """El borrador ya existe: que el modelo falle en un reintento (cupo, caída) no debe perder la
    ejecución. Conserva el borrador anterior y lo deja en manos de una persona"""
    issue = f"No se pudo generar un nuevo borrador ({describe_error(error)}); se muestra el anterior."
    update = {
        "generation_attempts": attempts,
        "generation_failed": True,
        "metrics": [step_metrics(f"generate{attempts}", start)],
        "audit": {"attempt": attempts, "decision": "conservar_borrador", "error": type(error).__name__},
    }
    # Fuente "human": la persona acaba de pedir esta corrección y aún no está en el borrador.
    # Si la validación rechazó después, las correcciones ya están aplicadas y se conservan
    if verdict.source == "human":
        issue += " La última corrección no se aplicó."
        update["human_corrections"] = state["human_corrections"][:-1]
    update["verdict"] = verdict.model_copy(update={"issues": [*verdict.issues, issue]})
    return update


RESTORED_SECTION_NAMES = {"education": "la educación", "languages": "los idiomas"}


def review_markdown(draft: StandardCV, lang: str, source: StandardCV | None = None) -> str:
    """El CV tal como lo ven los evaluadores: enmascarado y sin datos de contacto (no se evalúan).
    Con `source` usa los marcadores de ese CV: los del contexto y los que se restauran después"""
    mapping = mask_cv(source)[1] if source else None
    masked = mask_cv(draft, mapping)[0].model_copy(update={
        "email": None, "phone": None, "location": None, "linkedin_url": None, "portfolio_url": None,
    })
    return cv_to_markdown(masked, lang=lang)


def validate(state: AgentState, ctx: AgentContext) -> dict:
    """Valida de la capa más barata a la más cara: chequeos deterministas y luego modelo de decisión"""
    start = time.perf_counter()
    plan_result = state["plan"]
    corrections = state.get("human_corrections", [])
    issues = run_checks(state["draft_cv"], state["parsed_cv"], plan_result.output_language, plan_result.translate,
                        confirmed=corrections)
    decision_model = None

    if issues:
        verdict = Verdict(status="rejected", source="checks", issues=issues,
                          feedback=scrub_pii("\n".join(f"  - {issue}" for issue in issues)))
    else:
        try:
            result = evaluate_draft(state["context"], state["job_reqs"].must_have,
                                    review_markdown(state["draft_cv"], plan_result.output_language, state["parsed_cv"]),
                                    ctx.settings)
            decision_model = result.model
            status = status_from_scores(result.scores, human_corrected=bool(corrections))
            flagged = [b["feedback"] for name, b in BEHAVIORS.items()
                       if status == "rejected" and result.scores[name] >= 0.5]
            verdict = Verdict(status=status, source="decision_model", scores=result.scores,
                              feedback="\n".join(f"  - {f}" for f in flagged) or None)
        except DecisionError as error:
            # Sin modelo de decisión la validación no se detiene: decide el juez
            verdict = Verdict(status="uncertain", source="none", issues=[f"Modelo de decisión no disponible: {error}"])

    return {
        "verdict": verdict,
        "metrics": [step_metrics("validate", start)],
        "audit": {
            "attempt": state.get("generation_attempts", 0),
            "verdict": verdict.status,
            "source": verdict.source,
            "scores": verdict.scores,
            "issues": len(verdict.issues),
            "model": decision_model,
        },
    }


def judge(state: AgentState, ctx: AgentContext) -> dict:
    """Última capa automática: un modelo de lenguaje revisa lo que quedó sin resolver"""
    start = time.perf_counter()
    previous = state["verdict"]
    raw = None
    chain = JUDGE_PROMPT | get_structured_model("escalation", JudgeReview, include_raw=True, settings=ctx.settings)
    try:
        review, raw = invoke_structured(chain, {
            "context": state["context"],
            "requirements": ", ".join(state["job_reqs"].must_have) or "sin requisitos",
            "adapted_cv": review_markdown(state["draft_cv"], state["plan"].output_language, state["parsed_cv"]),
        }, "JudgeReview")
        status = "approved" if review.approved else "rejected"
        feedback = None
        if not review.approved:
            feedback = scrub_pii(review.feedback or "\n".join(review.issues)) or None
        verdict = Verdict(status=status, source="judge", scores=previous.scores, issues=review.issues, feedback=feedback)
        error_name = None
    except Exception as error:
        # El CV ya existe: que el juez falle (cupo, caída) no debe perder la ejecución
        verdict = previous.model_copy(update={"status": "uncertain", "issues": [*previous.issues, describe_error(error)]})
        error_name = type(error).__name__

    metrics = step_metrics("judge", start, raw)
    return {
        "verdict": verdict,
        "metrics": [metrics],
        "audit": {
            "attempt": state.get("generation_attempts", 0),
            "verdict": verdict.status,
            "model": metrics.model,
            "input_tokens": metrics.input_tokens,
            "output_tokens": metrics.output_tokens,
            "issues": len(verdict.issues),
            "error": error_name,
        },
    }


def _ask_human(kind: str, question: str, options: list[str], state: AgentState, feedback_allowed: bool = True) -> dict:
    """Pausa el grafo hasta recibir la decisión. El estado queda en Postgres, así que se puede
    retomar después, incluso desde otro proceso. Al retomar, el nodo se ejecuta otra vez desde el
    inicio: nada antes de esta llamada debe tener efectos"""
    verdict = state["verdict"]
    return interrupt({
        "kind": kind,
        "question": question,
        "options": options,
        "preview": cv_to_markdown(state["draft_cv"], lang=state["plan"].output_language),
        "status": verdict.status,
        "source": verdict.source,
        "scores": verdict.scores,
        "issues": verdict.issues,
        "warnings": state.get("warnings", []),
        "attempts": state.get("generation_attempts", 0),
        "corrections": state.get("human_corrections", []),
    })


def _human_correction(state: AgentState, text: str) -> dict:
    """Suma la instrucción a las anteriores, sin perderlas. Va a los modelos: se oculta lo
    personal que la persona haya escrito"""
    cleaned = scrub_pii(text).replace(state["parsed_cv"].candidate_name, CANDIDATE_TOKEN)
    verdict = state["verdict"]
    return {
        "human_corrections": [*state.get("human_corrections", []), cleaned],
        "verdict": Verdict(status="rejected", source="human", scores=verdict.scores, issues=verdict.issues),
    }


def _human_audit(kind: str, state: AgentState, answer: dict) -> dict:
    # El texto de la persona no se registra: puede contener datos personales
    verdict = state["verdict"]
    return {"kind": kind, "decision": answer["action"], "feedback_given": bool(answer.get("feedback")),
            "automatic_verdict": verdict.status, "automatic_source": verdict.source, "scores": verdict.scores}


def human_review(state: AgentState, ctx: AgentContext) -> dict:
    """Una persona revisa el CV ya aprobado antes de guardarlo"""
    answer = _ask_human("review", "El CV pasó la validación. ¿Lo aprueba?", REVIEW_OPTIONS, state)
    update = {"human_decision": answer["action"], "audit": _human_audit("review", state, answer)}
    if answer["action"] == "corregir":
        update.update(_human_correction(state, answer["feedback"]))
    return update


def human_escalation(state: AgentState, ctx: AgentContext) -> dict:
    """Último nivel de la validación: la automatización no pudo aprobar el CV"""
    resets = state.get("escalation_resets", 0)
    # Corregir da más intentos automáticos solo una vez, para no entrar en un bucle sin fin
    options = ESCALATION_OPTIONS if resets < MAX_ESCALATION_RESETS else [o for o in ESCALATION_OPTIONS if o != "corregir"]
    answer = _ask_human("escalation", "La validación automática no aprobó el CV. ¿Qué desea hacer?", options, state)
    update = {"human_decision": answer["action"], "audit": _human_audit("escalation", state, answer)}
    if answer["action"] == "corregir":
        update.update(_human_correction(state, answer["feedback"]), generation_attempts=0,
                      draft_repeated=False, escalation_resets=resets + 1)
    elif answer["action"] == "aceptar":
        verdict = state["verdict"]
        detail = "; ".join(verdict.issues) or "sin detalle"
        update["warnings"] = [f"Guardado por decisión humana, sin aprobación automática ({verdict.status}): {detail}"]
    return update


def render_and_save(state: AgentState, ctx: AgentContext) -> dict:
    plan_result, job_reqs = state["plan"], state["job_reqs"]
    lang = plan_result.output_language
    markdown = cv_to_markdown(state["draft_cv"], lang=lang)
    html = cv_to_html(state["draft_cv"], lang=lang)
    paths = save_outputs(state["candidate_id"], markdown, html, ctx.settings.output_dir, state["job_text"])

    warnings = []
    corrections = state.get("human_corrections", [])
    if corrections:
        warnings.append("El CV incluye cambios indicados por una persona, que la validación aceptó como ciertos: "
                        + "; ".join(corrections))
    try:
        memory.save_generation(get_store(), state["candidate_id"], scrub_pii(state["job_text"]), markdown, {
            "job_title": job_reqs.title,
            "language": lang,
            "skill_coverage": plan_result.skill_coverage,
            "missing_requirements": plan_result.missing_requirements,
            "verdict": state["verdict"].status,
            "verdict_scores": state["verdict"].scores,
            "human_decision": state.get("human_decision"),
            "human_corrections": len(corrections),
            "generation_attempts": state.get("generation_attempts", 0),
            "files": [str(p) for p in paths],
            "thread_id": ctx.thread_id,
        })
    except Exception as error:
        # El CV ya está guardado en disco: perder el registro no invalida la ejecución
        warnings.append(f"No se pudo registrar la generación en la memoria ({type(error).__name__}).")

    return {
        "markdown": markdown,
        "html": html,
        "output_paths": [str(p) for p in paths],
        "warnings": warnings,
        # Los nombres de archivo llevan el nombre del candidato: solo se registran las extensiones
        "audit": {"files": [p.suffix for p in paths], "language": lang},
    }
