"""Estado del grafo y contexto de ejecución (recursos que no forman parte del estado)."""

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from cvertidor.config import Settings
from cvertidor.graph.planning import Plan
from cvertidor.llm.embeddings import EmbeddingsInfo, get_embeddings
from cvertidor.observability import AuditLog
from cvertidor.pipeline import StepMetrics
from cvertidor.rag.vector_store import SyncStats, get_vector_store
from cvertidor.schemas.cv import StandardCV
from cvertidor.schemas.evaluation import Verdict
from cvertidor.schemas.job import JobRequirements


def accumulate(current: list | None, update: list | None) -> list:
    """Acumula entre nodos. El estado persiste por hilo: None lo reinicia al empezar otra ejecución"""
    if update is None:
        return []
    return (current or []) + update


class AgentState(TypedDict, total=False):
    # Entrada: "run" genera un CV; "chat" conversa sobre el que ya existe en el hilo
    mode: Literal["run", "chat"]
    cv_text: str
    job_text: str
    run_id: str
    # Omite la revisión humana de un CV aprobado (la escalación siempre pregunta)
    auto_approve: bool
    # load_memory: qué se reutilizó de la memoria de largo plazo
    runs_in_thread: int
    cv_hash: str
    job_hash: str
    cv_from_memory: bool
    job_from_memory: bool
    # ingest_cv e index_cv
    candidate_id: str
    parsed_cv: StandardCV
    sync: SyncStats
    # analyze_job y plan
    job_reqs: JobRequirements
    preferences: list[str]
    similar_generations: list[dict]
    plan: Plan
    # retrieve (puede repetirse con consultas y umbrales más amplios)
    retrieval_attempts: int
    retrieved_docs: list[Document]
    # generate, validate y judge (el bucle de corrección)
    generation_attempts: int
    # El intento repitió el borrador anterior: otro reintento no cambiaría nada
    draft_repeated: bool
    # El modelo falló en un reintento: se conserva el borrador anterior y decide una persona
    generation_failed: bool
    context: str
    draft_cv: StandardCV
    verdict: Verdict
    # human_review y human_escalation
    human_decision: str | None
    # Se acumulan durante la ejecución, enmascaradas. No pasan al perfil de largo plazo
    human_corrections: list[str]
    escalation_resets: int
    # render_and_save
    markdown: str
    html: str
    output_paths: list[str]
    # Conversación (memoria de corto plazo). Los mensajes llegan enmascarados a los modelos
    messages: Annotated[list[AnyMessage], add_messages]
    conversation_summary: str | None
    refine_steps: int
    # Preferencias que el modelo propuso y que la persona debe confirmar
    preference_queue: Annotated[list[dict], accumulate]
    preferences_handled: int
    # Se acumulan entre nodos (ingest_cv y analyze_job corren en paralelo)
    warnings: Annotated[list[str], accumulate]
    metrics: Annotated[list[StepMetrics], accumulate]


@dataclass
class AgentContext:
    """Recursos de la ejecución que no van en el estado (clientes, conexiones, callbacks)"""
    settings: Settings
    audit: AuditLog
    on_step: Callable[[str], None] = lambda message: None
    embeddings_info: EmbeddingsInfo | None = None
    # Ignora los resultados guardados y vuelve a llamar a los modelos
    reprocess: bool = False
    thread_id: str | None = None
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
