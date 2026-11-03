"""Funciones de enrutamiento del grafo (decisiones adaptativas entre nodos)."""

from typing import Literal

from langchain_core.documents import Document

from cvertidor.graph.state import AgentState

# Intentos totales de recuperación: el inicial más dos más amplios
MAX_RETRIEVAL_ATTEMPTS = 3
# Responsabilidades relevantes mínimas para armar la sección de experiencia
# (cada responsabilidad es un chunk; con una sola, la sección queda en un único punto)
MIN_WORK_HISTORY_CHUNKS = 2


def has_work_history(docs: list[Document]) -> bool:
    return sum(d.metadata.get("section") == "work_history" for d in docs) >= MIN_WORK_HISTORY_CHUNKS


def route_after_retrieve(state: AgentState) -> Literal["retrieve", "generate"]:
    """Sin experiencia laboral relevante se reintenta con una consulta y un umbral más
    amplios; al agotar los intentos se genera igual (con un aviso)"""
    if has_work_history(state.get("retrieved_docs", [])):
        return "generate"
    if state.get("retrieval_attempts", 0) >= MAX_RETRIEVAL_ATTEMPTS:
        return "generate"
    return "retrieve"
