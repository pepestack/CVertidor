"""Funciones de enrutamiento del grafo (decisiones adaptativas entre nodos)."""

from typing import Literal

from langchain_core.documents import Document

from cvertidor.graph.state import AgentState

# Intentos totales de recuperación: el inicial más dos más amplios
MAX_RETRIEVAL_ATTEMPTS = 3
# Responsabilidades relevantes mínimas para armar la sección de experiencia
# (cada responsabilidad es un chunk; con una sola, la sección queda en un único punto)
MIN_WORK_HISTORY_CHUNKS = 2
# Intentos totales de generación: el inicial más dos con correcciones
MAX_GENERATION_ATTEMPTS = 3


def has_work_history(docs: list[Document]) -> bool:
    return sum(d.metadata.get("section") == "work_history" for d in docs) >= MIN_WORK_HISTORY_CHUNKS


def route_entry(state: AgentState) -> Literal["refine_agent", "load_memory"]:
    return "refine_agent" if state.get("mode") == "chat" else "load_memory"


def route_after_refine(state: AgentState) -> Literal["tools", "__end__"]:
    return "tools" if state["messages"][-1].tool_calls else "__end__"


def route_after_tools(state: AgentState) -> Literal["confirm_preference", "refine_agent"]:
    """Toda preferencia propuesta pasa por la persona antes de que el agente continúe"""
    if len(state.get("preference_queue") or []) > state.get("preferences_handled", 0):
        return "confirm_preference"
    return "refine_agent"


def route_after_memory(state: AgentState) -> list[str]:
    """Solo se llama a los modelos para lo que la memoria no tiene: un CV sin cambios pasa
    directo a indexar y una oferta ya analizada no se vuelve a analizar"""
    routes = ["index_cv" if state.get("cv_from_memory") else "ingest_cv"]
    if not state.get("job_from_memory"):
        routes.append("analyze_job")
    return routes


def route_after_retrieve(state: AgentState) -> Literal["retrieve", "generate"]:
    """Sin experiencia laboral relevante se reintenta con una consulta y un umbral más
    amplios; al agotar los intentos se genera igual (con un aviso)"""
    if has_work_history(state.get("retrieved_docs", [])):
        return "generate"
    if state.get("retrieval_attempts", 0) >= MAX_RETRIEVAL_ATTEMPTS:
        return "generate"
    return "retrieve"


def route_after_generate(state: AgentState) -> Literal["validate", "human_escalation"]:
    """Sin un borrador nuevo no se valida otra vez: gastaría una llamada y podría aprobar el
    mismo borrador que se acaba de rechazar"""
    return "human_escalation" if state.get("generation_failed") else "validate"


def retries_left(state: AgentState) -> bool:
    return state.get("generation_attempts", 0) < MAX_GENERATION_ATTEMPTS and not state.get("draft_repeated")


def route_by_verdict(state: AgentState) -> Literal["human_review", "render_and_save", "generate", "human_escalation"]:
    """Aprobado pasa a la revisión humana; rechazado se corrige mientras queden intentos.
    Lo que la automatización no resuelve siempre lo decide una persona, incluso con --auto"""
    status = state["verdict"].status
    if status == "approved":
        return "render_and_save" if state.get("auto_approve") else "human_review"
    if status == "rejected" and retries_left(state):
        return "generate"
    return "human_escalation"


def route_after_validate(state: AgentState) -> str:
    """Un resultado dudoso va primero al juez"""
    if state["verdict"].status == "uncertain":
        return "judge"
    return route_by_verdict(state)


def route_after_human(state: AgentState) -> Literal["render_and_save", "generate", "__end__"]:
    decision = state["human_decision"]
    if decision == "corregir":
        return "generate"
    return "__end__" if decision == "descartar" else "render_and_save"
