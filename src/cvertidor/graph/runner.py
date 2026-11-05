"""Ejecución del grafo completo: valida la entrada, prepara el contexto e invoca el grafo."""

import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from cvertidor.config import Settings, get_settings
from cvertidor.graph.builder import build_graph
from cvertidor.graph.planning import Plan
from cvertidor.graph.state import AgentContext
from cvertidor.ingestion.pii import scrub_pii, unmask
from cvertidor.llm.embeddings import EmbeddingsInfo, get_embeddings
from cvertidor.llm.providers import message_text
from cvertidor.memory.checkpointer import open_checkpointer
from cvertidor.memory.store import open_store
from cvertidor.observability import AuditLog
from cvertidor.pipeline import (
    CANDIDATE_TOKEN,
    InputError,
    MIN_CV_LENGTH,
    MIN_JOB_DESC_LENGTH,
    StepMetrics,
    ensure_db_available,
    mask_cv,
    validate_text,
)
from cvertidor.rag.vector_store import SyncStats
from cvertidor.schemas.cv import StandardCV
from cvertidor.schemas.evaluation import Verdict
from cvertidor.schemas.job import JobRequirements


@dataclass
class AgentResult:
    run_id: str
    thread_id: str
    runs_in_thread: int
    cv_from_memory: bool
    job_from_memory: bool
    preferences: list[str]
    similar_generations: list[dict]
    candidate_id: str
    parsed_cv: StandardCV
    job_reqs: JobRequirements
    plan: Plan
    adapted_cv: StandardCV
    verdict: Verdict
    generation_attempts: int
    human_decision: str | None
    human_corrections: list[str]
    discarded: bool
    markdown: str
    html: str
    output_paths: list[Path]
    retrieved_docs: list[Document]
    retrieval_attempts: int
    sync: SyncStats
    embeddings_model: str
    warnings: list[str] = field(default_factory=list)
    metrics: list[StepMetrics] = field(default_factory=list)
    audit_events: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PendingDecision:
    """El grafo está pausado esperando a una persona; se continúa con resume_agent()"""
    thread_id: str
    run_id: str
    kind: str
    question: str
    options: list[str]
    preview: str
    status: str
    source: str
    scores: dict[str, float] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    attempts: int = 0
    corrections: list[str] = field(default_factory=list)


@dataclass
class ChatReply:
    """Respuesta del agente a un mensaje; los archivos son solo los guardados en este turno"""
    thread_id: str
    run_id: str
    text: str
    output_paths: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: list[StepMetrics] = field(default_factory=list)


@contextmanager
def _open_graph(db_url: str, info: EmbeddingsInfo) -> Iterator[Any]:
    with open_checkpointer(db_url) as checkpointer, open_store(db_url, info) as store:
        yield build_graph(checkpointer, store)


def _thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _outcome(state: dict, ctx: AgentContext) -> "AgentResult | PendingDecision | ChatReply":
    interrupts = state.get("__interrupt__")
    if interrupts:
        return PendingDecision(thread_id=ctx.thread_id, run_id=ctx.audit.run_id, **interrupts[0].value)
    if state.get("mode") == "chat":
        # El modelo solo ve marcadores: la persona lee sus datos reales
        return ChatReply(
            thread_id=ctx.thread_id,
            run_id=ctx.audit.run_id,
            text=unmask(message_text(state["messages"][-1]), mask_cv(state["parsed_cv"])[1]),
            output_paths=[Path(p) for p in state.get("output_paths", [])],
            warnings=state.get("warnings", []),
            metrics=state.get("metrics", []),
        )
    decision = state.get("human_decision")
    return AgentResult(
        run_id=ctx.audit.run_id,
        thread_id=ctx.thread_id,
        runs_in_thread=state["runs_in_thread"],
        cv_from_memory=state["cv_from_memory"],
        job_from_memory=state["job_from_memory"],
        preferences=state["preferences"],
        similar_generations=state["similar_generations"],
        candidate_id=state["candidate_id"],
        parsed_cv=state["parsed_cv"],
        job_reqs=state["job_reqs"],
        plan=state["plan"],
        adapted_cv=state["draft_cv"],
        verdict=state["verdict"],
        generation_attempts=state["generation_attempts"],
        human_decision=decision,
        human_corrections=state.get("human_corrections", []),
        discarded=decision == "descartar",
        markdown=state.get("markdown", ""),
        html=state.get("html", ""),
        output_paths=[Path(p) for p in state.get("output_paths", [])],
        retrieved_docs=state["retrieved_docs"],
        retrieval_attempts=state["retrieval_attempts"],
        sync=state["sync"],
        embeddings_model=ctx.embeddings().model,
        warnings=state.get("warnings", []),
        metrics=state.get("metrics", []),
        audit_events=ctx.audit.events,
    )


def run_agent(
    cv_text: str,
    job_text: str,
    settings: Settings | None = None,
    embeddings_info: EmbeddingsInfo | None = None,
    on_step: Callable[[str], None] = lambda message: None,
    thread_id: str | None = None,
    reprocess: bool = False,
    auto_approve: bool = False,
) -> "AgentResult | PendingDecision":
    settings = settings or get_settings()
    cv_text = validate_text(cv_text, MIN_CV_LENGTH, "El CV")
    job_text = validate_text(job_text, MIN_JOB_DESC_LENGTH, "La oferta de trabajo")
    db_url = settings.require_db_url()
    # Falla antes de gastar llamadas a modelos si la base no responde
    ensure_db_available(db_url)

    run_id = uuid.uuid4().hex[:12]
    thread_id = thread_id or f"hilo-{run_id}"
    # El store necesita los embeddings para su índice semántico
    info = embeddings_info or get_embeddings(settings)
    ctx = AgentContext(settings=settings, audit=AuditLog(settings.log_dir, run_id, thread_id), on_step=on_step,
                       embeddings_info=info, reprocess=reprocess, thread_id=thread_id)

    with _open_graph(db_url, info) as graph:
        if graph.get_state(_thread_config(thread_id)).interrupts:
            raise InputError(f"El hilo '{thread_id}' tiene una decisión pendiente. Retómelo con: cvertidor resume --thread {thread_id}")
        # None reinicia los acumuladores: el hilo conserva el estado de sus ejecuciones anteriores
        state = graph.invoke(
            {"mode": "run", "cv_text": cv_text, "job_text": job_text, "run_id": run_id, "auto_approve": auto_approve,
             "warnings": None, "metrics": None},
            config=_thread_config(thread_id),
            context=ctx,
        )
    return _outcome(state, ctx)


def chat_agent(
    thread_id: str,
    message: str,
    settings: Settings | None = None,
    embeddings_info: EmbeddingsInfo | None = None,
    on_step: Callable[[str], None] = lambda message: None,
) -> "ChatReply | PendingDecision":
    """Un mensaje de la persona sobre el CV que ya existe en el hilo"""
    settings = settings or get_settings()
    message = validate_text(message, 1, "El mensaje")
    db_url = settings.require_db_url()
    ensure_db_available(db_url)
    info = embeddings_info or get_embeddings(settings)

    with _open_graph(db_url, info) as graph:
        config = _thread_config(thread_id)
        snapshot = graph.get_state(config)
        if snapshot.interrupts:
            raise InputError(f"El hilo '{thread_id}' tiene una decisión pendiente. Retómelo con: cvertidor resume --thread {thread_id}")
        if not snapshot.values.get("draft_cv"):
            raise InputError(f"El hilo '{thread_id}' no tiene un CV. Genérelo con: cvertidor run --thread {thread_id} --cv ... --job ...")

        run_id = uuid.uuid4().hex[:12]
        ctx = AgentContext(settings=settings, audit=AuditLog(settings.log_dir, run_id, thread_id), on_step=on_step,
                           embeddings_info=info, thread_id=thread_id)
        # El mensaje va a los modelos: se oculta lo personal antes de que entre al historial
        name = snapshot.values["parsed_cv"].candidate_name
        cleaned = scrub_pii(message).replace(name, CANDIDATE_TOKEN)
        state = graph.invoke(
            {"mode": "chat", "run_id": run_id, "messages": [HumanMessage(cleaned)], "refine_steps": 0,
             "preference_queue": None, "preferences_handled": 0, "output_paths": [],
             "warnings": None, "metrics": None},
            config=config,
            context=ctx,
        )
    return _outcome(state, ctx)


def pending_decision(thread_id: str, settings: Settings | None = None,
                     embeddings_info: EmbeddingsInfo | None = None) -> PendingDecision:
    """La decisión que espera un hilo pausado, sin continuarlo"""
    settings = settings or get_settings()
    db_url = settings.require_db_url()
    ensure_db_available(db_url)
    with _open_graph(db_url, embeddings_info or get_embeddings(settings)) as graph:
        snapshot = graph.get_state(_thread_config(thread_id))
    if not snapshot.interrupts:
        raise InputError(f"El hilo '{thread_id}' no tiene ninguna decisión pendiente.")
    return PendingDecision(thread_id=thread_id, run_id=snapshot.values.get("run_id", ""), **snapshot.interrupts[0].value)


def resume_agent(
    thread_id: str,
    action: str,
    feedback: str | None = None,
    settings: Settings | None = None,
    embeddings_info: EmbeddingsInfo | None = None,
    on_step: Callable[[str], None] = lambda message: None,
) -> "AgentResult | PendingDecision":
    """Continúa un hilo pausado con la decisión de la persona. Funciona desde otro proceso:
    el estado y la pregunta pendiente se leen de Postgres"""
    settings = settings or get_settings()
    db_url = settings.require_db_url()
    ensure_db_available(db_url)
    info = embeddings_info or get_embeddings(settings)

    with _open_graph(db_url, info) as graph:
        config = _thread_config(thread_id)
        snapshot = graph.get_state(config)
        if not snapshot.interrupts:
            raise InputError(f"El hilo '{thread_id}' no tiene ninguna decisión pendiente.")
        options = snapshot.interrupts[0].value["options"]
        if action not in options:
            raise InputError(f"Decisión no válida: '{action}'. Opciones: {', '.join(options)}")
        if action == "corregir" and not (feedback or "").strip():
            raise InputError("Para corregir hay que indicar qué cambiar.")

        run_id = snapshot.values.get("run_id") or uuid.uuid4().hex[:12]
        ctx = AgentContext(settings=settings, audit=AuditLog(settings.log_dir, run_id, thread_id), on_step=on_step,
                           embeddings_info=info, thread_id=thread_id)
        state = graph.invoke(Command(resume={"action": action, "feedback": feedback}), config=config, context=ctx)
    return _outcome(state, ctx)
