"""Construcción del grafo de LangGraph.

    START ─┬─> ingest_cv ───┐
           └─> analyze_job ─┴─> plan ─> retrieve ─┬─> generate ─> render_and_save ─> END
                                          ^       │
                                          └───────┘ sin experiencia relevante (máx. 3 intentos)
"""

import time
from collections.abc import Callable

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from cvertidor.graph import nodes
from cvertidor.graph.edges import route_after_retrieve
from cvertidor.graph.state import AgentContext, AgentState

NodeFn = Callable[[AgentState, AgentContext], dict]

# Mensaje de progreso de cada nodo; el de retrieve cambia en los reintentos
STEP_LABELS: dict[str, Callable[[AgentState], str]] = {
    "ingest_cv": lambda s: "Extrayendo la estructura del CV (datos personales enmascarados) e indexándolo en pgvector...",
    "analyze_job": lambda s: "Analizando los requisitos de la oferta...",
    "plan": lambda s: "Planificando la adaptación (idioma, prioridades y consultas)...",
    "retrieve": lambda s: (
        "Recuperando la experiencia relevante para la oferta..." if not s.get("retrieval_attempts")
        else f"Sin experiencia relevante: reintentando con una consulta más amplia (intento {s['retrieval_attempts'] + 1})..."
    ),
    "generate": lambda s: "Generando el CV adaptado a la oferta...",
    "render_and_save": lambda s: "Guardando el CV en Markdown y HTML...",
}


def audited(name: str, fn: NodeFn):
    """Envuelve un nodo: informa el progreso, mide el tiempo y registra el evento de auditoría"""
    def node(state: AgentState, runtime: Runtime[AgentContext]) -> dict:
        ctx = runtime.context
        ctx.on_step(STEP_LABELS[name](state))
        start = time.perf_counter()
        try:
            update = fn(state, ctx)
        except Exception as error:
            ctx.audit.record(name, status="error", seconds=round(time.perf_counter() - start, 3),
                             error=type(error).__name__)
            raise
        details = update.pop("audit", {})
        ctx.audit.record(name, status="ok", seconds=round(time.perf_counter() - start, 3), **details)
        return update

    node.__name__ = name
    return node


def build_graph(checkpointer=None, store=None):
    builder = StateGraph(AgentState, context_schema=AgentContext)
    for name in ("ingest_cv", "analyze_job", "plan", "retrieve", "generate", "render_and_save"):
        builder.add_node(name, audited(name, getattr(nodes, name)))

    # El CV y la oferta se analizan en paralelo; plan espera a ambos
    builder.add_edge(START, "ingest_cv")
    builder.add_edge(START, "analyze_job")
    builder.add_edge(["ingest_cv", "analyze_job"], "plan")
    builder.add_edge("plan", "retrieve")
    builder.add_conditional_edges("retrieve", route_after_retrieve, ["retrieve", "generate"])
    builder.add_edge("generate", "render_and_save")
    builder.add_edge("render_and_save", END)
    return builder.compile(checkpointer=checkpointer, store=store)


def graph_mermaid() -> str:
    return build_graph().get_graph().draw_mermaid()
