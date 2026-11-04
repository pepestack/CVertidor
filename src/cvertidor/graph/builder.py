"""Construcción del grafo de LangGraph.

    START ─> load_memory ─┬─> [CV nuevo] ingest_cv ─┬─> index_cv ─┐
                          ├─> [CV conocido] ────────┘             ├─> plan ─> retrieve ─┬─> generate ─> render_and_save ─> END
                          └─> [oferta nueva] analyze_job ─────────┘             ^       │
                                                                                └───────┘ sin experiencia relevante (máx. 3 intentos)

    generate ─> validate ─┬─> [aprobado] render_and_save
       ^                  ├─> [dudoso] judge ─┬─> [aprobado] render_and_save
       │                  │                   │
       └── [rechazado, máx. 3 intentos] ──────┘
"""

import time
from collections.abc import Callable

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from cvertidor.graph import nodes
from cvertidor.graph.edges import route_after_judge, route_after_memory, route_after_retrieve, route_after_validate
from cvertidor.graph.state import AgentContext, AgentState

NodeFn = Callable[[AgentState, AgentContext], dict]

# Mensaje de progreso de cada nodo; el de retrieve cambia en los reintentos
STEP_LABELS: dict[str, Callable[[AgentState], str]] = {
    "load_memory": lambda s: "Consultando la memoria del agente...",
    "ingest_cv": lambda s: "Extrayendo la estructura del CV (datos personales enmascarados)...",
    "index_cv": lambda s: "Sincronizando los fragmentos del CV en pgvector...",
    "analyze_job": lambda s: "Analizando los requisitos de la oferta...",
    "plan": lambda s: "Planificando la adaptación (idioma, prioridades y consultas)...",
    "retrieve": lambda s: (
        "Recuperando la experiencia relevante para la oferta..." if not s.get("retrieval_attempts")
        else f"Sin experiencia relevante: reintentando con una consulta más amplia (intento {s['retrieval_attempts'] + 1})..."
    ),
    "generate": lambda s: (
        "Generando el CV adaptado a la oferta..." if not s.get("generation_attempts")
        else f"Corrigiendo el CV según la validación (intento {s['generation_attempts'] + 1})..."
    ),
    "validate": lambda s: "Validando el CV generado...",
    "judge": lambda s: "Resultado dudoso: consultando al juez...",
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
    for name in ("load_memory", "ingest_cv", "index_cv", "analyze_job", "retrieve", "generate", "validate", "judge",
                 "render_and_save"):
        builder.add_node(name, audited(name, getattr(nodes, name)))
    # defer: plan espera a que terminen las ramas que se hayan activado (pueden ser una o dos)
    builder.add_node("plan", audited("plan", nodes.plan), defer=True)

    builder.add_edge(START, "load_memory")
    builder.add_conditional_edges("load_memory", route_after_memory, ["ingest_cv", "index_cv", "analyze_job"])
    builder.add_edge("ingest_cv", "index_cv")
    builder.add_edge("index_cv", "plan")
    builder.add_edge("analyze_job", "plan")
    builder.add_edge("plan", "retrieve")
    builder.add_conditional_edges("retrieve", route_after_retrieve, ["retrieve", "generate"])
    builder.add_edge("generate", "validate")
    builder.add_conditional_edges("validate", route_after_validate, ["render_and_save", "generate", "judge"])
    builder.add_conditional_edges("judge", route_after_judge, ["render_and_save", "generate"])
    builder.add_edge("render_and_save", END)
    return builder.compile(checkpointer=checkpointer, store=store)


def graph_mermaid() -> str:
    return build_graph().get_graph().draw_mermaid()
