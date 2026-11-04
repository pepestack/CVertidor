"""Ejecución del grafo completo: valida la entrada, prepara el contexto e invoca el grafo."""

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from cvertidor.config import Settings, get_settings
from cvertidor.graph.builder import build_graph
from cvertidor.graph.planning import Plan
from cvertidor.graph.state import AgentContext
from cvertidor.llm.embeddings import EmbeddingsInfo, get_embeddings
from cvertidor.memory.checkpointer import open_checkpointer
from cvertidor.memory.store import open_store
from cvertidor.observability import AuditLog
from cvertidor.pipeline import (
    MIN_CV_LENGTH,
    MIN_JOB_DESC_LENGTH,
    StepMetrics,
    ensure_db_available,
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


def run_agent(
    cv_text: str,
    job_text: str,
    settings: Settings | None = None,
    embeddings_info: EmbeddingsInfo | None = None,
    on_step: Callable[[str], None] = lambda message: None,
    thread_id: str | None = None,
    reprocess: bool = False,
) -> AgentResult:
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

    with open_checkpointer(db_url) as checkpointer, open_store(db_url, info) as store:
        graph = build_graph(checkpointer, store)
        # None reinicia los acumuladores: el hilo conserva el estado de sus ejecuciones anteriores
        state = graph.invoke(
            {"cv_text": cv_text, "job_text": job_text, "warnings": None, "metrics": None},
            config={"configurable": {"thread_id": thread_id}},
            context=ctx,
        )

    return AgentResult(
        run_id=run_id,
        thread_id=thread_id,
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
        markdown=state["markdown"],
        html=state["html"],
        output_paths=[Path(p) for p in state["output_paths"]],
        retrieved_docs=state["retrieved_docs"],
        retrieval_attempts=state["retrieval_attempts"],
        sync=state["sync"],
        embeddings_model=info.model,
        warnings=state.get("warnings", []),
        metrics=state.get("metrics", []),
        audit_events=ctx.audit.events,
    )
