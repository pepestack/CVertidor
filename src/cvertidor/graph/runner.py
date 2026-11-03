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
from cvertidor.llm.embeddings import EmbeddingsInfo
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
from cvertidor.schemas.job import JobRequirements


@dataclass
class AgentResult:
    run_id: str
    candidate_id: str
    parsed_cv: StandardCV
    job_reqs: JobRequirements
    plan: Plan
    adapted_cv: StandardCV
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
    graph=None,
) -> AgentResult:
    settings = settings or get_settings()
    cv_text = validate_text(cv_text, MIN_CV_LENGTH, "El CV")
    job_text = validate_text(job_text, MIN_JOB_DESC_LENGTH, "La oferta de trabajo")
    # Falla antes de gastar llamadas a modelos si la base no responde
    ensure_db_available(settings.require_db_url())

    run_id = uuid.uuid4().hex[:12]
    ctx = AgentContext(settings=settings, audit=AuditLog(settings.log_dir, run_id),
                       on_step=on_step, embeddings_info=embeddings_info)
    graph = graph or build_graph()
    state = graph.invoke({"cv_text": cv_text, "job_text": job_text}, context=ctx)

    return AgentResult(
        run_id=run_id,
        candidate_id=state["candidate_id"],
        parsed_cv=state["parsed_cv"],
        job_reqs=state["job_reqs"],
        plan=state["plan"],
        adapted_cv=state["draft_cv"],
        markdown=state["markdown"],
        html=state["html"],
        output_paths=[Path(p) for p in state["output_paths"]],
        retrieved_docs=state["retrieved_docs"],
        retrieval_attempts=state["retrieval_attempts"],
        sync=state["sync"],
        embeddings_model=ctx.embeddings().model,
        warnings=state.get("warnings", []),
        metrics=state.get("metrics", []),
        audit_events=ctx.audit.events,
    )
