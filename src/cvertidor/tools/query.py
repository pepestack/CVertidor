"""Herramientas de consulta del agente de refinamiento.

El candidate_id nunca es un argumento: se toma del estado del grafo. Así, una instrucción
maliciosa en un CV o una oferta no puede hacer que el modelo lea datos de otro candidato.
Todo lo que devuelven va enmascarado, porque vuelve al modelo.
"""

from typing import Any

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

# LangGraph inyecta este parámetro: no forma parte de los argumentos que ve el modelo.
# Sin parámetros de tipo, Pydantic asume que no hay contexto y emite una advertencia
Runtime = ToolRuntime[Any, Any]


def record_tool(runtime: Runtime, name: str, **details) -> None:
    runtime.context.on_step(f"Herramienta: {name}")
    runtime.context.audit.record(f"tool:{name}", status="ok", **details)


@tool
def search_candidate_experience(query: str, runtime: Runtime) -> str:
    """Busca en el CV del candidato los fragmentos más relevantes para una consulta (búsqueda semántica)."""
    from cvertidor.graph import nodes

    ctx, state = runtime.context, runtime.state
    info = ctx.embeddings()
    threshold = nodes.similarity_threshold(info.slug) - nodes.RETRIEVAL_THRESHOLD_STEP
    retriever = nodes.build_retriever(ctx.vector_store(), info.embeddings, state["candidate_id"], threshold)
    docs = retriever.invoke(nodes.scrub_pii(query))
    record_tool(runtime, "search_candidate_experience", found=len(docs))
    return nodes.format_docs(docs) or "No se encontró nada relevante en el CV."


@tool
def get_candidate_profile(runtime: Runtime) -> str:
    """Devuelve el CV original completo del candidato, para consultar lo que el CV adaptado omitió."""
    from cvertidor.graph import nodes

    record_tool(runtime, "get_candidate_profile")
    return nodes.review_markdown(runtime.state["parsed_cv"], "es")


@tool
def recall_preferences(runtime: Runtime, query: str = "") -> str:
    """Consulta las preferencias recordadas del candidato. Con `query`, las más relacionadas primero."""
    from cvertidor.graph import nodes

    found = nodes.memory.search_preferences(runtime.store, runtime.state["candidate_id"], nodes.scrub_pii(query))
    record_tool(runtime, "recall_preferences", found=len(found))
    return "\n".join(f"- {text}" for text in found) or "No hay preferencias guardadas."
