"""Estado del grafo y contexto de ejecución (recursos que no forman parte del estado)."""

import operator
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langchain_core.documents import Document

from cvertidor.config import Settings
from cvertidor.graph.planning import Plan
from cvertidor.llm.embeddings import EmbeddingsInfo, get_embeddings
from cvertidor.observability import AuditLog
from cvertidor.pipeline import StepMetrics
from cvertidor.rag.vector_store import SyncStats, get_vector_store
from cvertidor.schemas.cv import StandardCV
from cvertidor.schemas.job import JobRequirements


class AgentState(TypedDict, total=False):
    # Entrada
    cv_text: str
    job_text: str
    # ingest_cv
    candidate_id: str
    parsed_cv: StandardCV
    sync: SyncStats
    # analyze_job y plan
    job_reqs: JobRequirements
    plan: Plan
    # retrieve (puede repetirse con consultas y umbrales más amplios)
    retrieval_attempts: int
    retrieved_docs: list[Document]
    # generate y render_and_save
    draft_cv: StandardCV
    markdown: str
    html: str
    output_paths: list[str]
    # Se acumulan entre nodos (ingest_cv y analyze_job corren en paralelo)
    warnings: Annotated[list[str], operator.add]
    metrics: Annotated[list[StepMetrics], operator.add]


@dataclass
class AgentContext:
    """Recursos de la ejecución que no van en el estado (clientes, conexiones, callbacks)"""
    settings: Settings
    audit: AuditLog
    on_step: Callable[[str], None] = lambda message: None
    embeddings_info: EmbeddingsInfo | None = None
    _vector_store: Any = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def embeddings(self) -> EmbeddingsInfo:
        with self._lock:
            if self.embeddings_info is None:
                self.embeddings_info = get_embeddings(self.settings)
            return self.embeddings_info

    def vector_store(self):
        info = self.embeddings()
        with self._lock:
            if self._vector_store is None:
                self._vector_store = get_vector_store(info.embeddings, info.slug, self.settings.require_db_url())
            return self._vector_store
