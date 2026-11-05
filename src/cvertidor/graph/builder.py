"""Construcción del grafo de LangGraph.

    START ─> load_memory ─┬─> [CV nuevo] ingest_cv ─┬─> index_cv ─┐
                          ├─> [CV conocido] ────────┘             ├─> plan ─> retrieve ─┬─> generate ─> render_and_save ─> END
                          └─> [oferta nueva] analyze_job ─────────┘             ^       │
                                                                                └───────┘ sin experiencia relevante (máx. 3 intentos)

    generate ─> validate ─┬─> [aprobado] human_review ⏸ ─┬─> [aprobar] render_and_save ─> END
       ^                  ├─> [dudoso] judge (decide igual) │
       ├── [rechazado, máx. 3 intentos]                     ├─> [corregir] generate
       │                  └─> [sin resolver] human_escalation ⏸ ─┬─> [aceptar] render_and_save
       └── [corregir, una vez] ──────────────────────────────────┤
                                                                 └─> [descartar] END

    Si el modelo falla en un reintento: generate ─> human_escalation ⏸, con el borrador anterior.

    Con --auto, un CV aprobado pasa directo a render_and_save.

    Conversación sobre un CV existente (cvertidor chat):
    START ─> refine_agent ⇄ tools ─> [preferencia propuesta] confirm_preference ⏸ ─> refine_agent ─> END
"""

import time
from collections.abc import Callable

from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from cvertidor.graph import nodes, refine
from cvertidor.graph.edges import (
    route_after_generate,
    route_after_human,
    route_after_memory,
    route_after_refine,
    route_after_retrieve,
    route_after_tools,
    route_after_validate,
    route_by_verdict,
    route_entry,
)
from cvertidor.graph.state import AgentContext, AgentState

NodeFn = Callable[[AgentState, AgentContext], dict]

# Mensaje de progreso de cada nodo; el de retrieve cambia en los reintentos. Los nodos humanos
# no tienen: muestran su pregunta, y al retomar se ejecutan de nuevo
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
        else f"Aplicando las correcciones de la persona (intento {s['generation_attempts'] + 1})..." if s.get("human_corrections")
        else f"Corrigiendo el CV según la validación (intento {s['generation_attempts'] + 1})..."
    ),
    "validate": lambda s: "Validando el CV generado...",
    "judge": lambda s: "Resultado dudoso: consultando al juez...",
    "render_and_save": lambda s: "Guardando el CV en Markdown y HTML...",
    "refine_agent": lambda s: "Pensando...",
}


def audited(name: str, fn: NodeFn):
    """Envuelve un nodo: informa el progreso, mide el tiempo y registra el evento de auditoría"""
    def node(state: AgentState, runtime: Runtime[AgentContext]) -> dict:
        ctx = runtime.context
        if name in STEP_LABELS:
            ctx.on_step(STEP_LABELS[name](state))
        start = time.perf_counter()
        try:
            update = fn(state, ctx)
        except GraphBubbleUp:
            # interrupt() pausa el grafo lanzando esta excepción: no es un error
            ctx.audit.record(name, status="paused", seconds=round(time.perf_counter() - start, 3))
            raise
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
                 "human_review", "human_escalation", "render_and_save"):
        builder.add_node(name, audited(name, getattr(nodes, name)))
    # defer: plan espera a que terminen las ramas que se hayan activado (pueden ser una o dos)
    builder.add_node("plan", audited("plan", nodes.plan), defer=True)

    builder.add_node("refine_agent", audited("refine_agent", refine.refine_agent))
    builder.add_node("confirm_preference", audited("confirm_preference", refine.confirm_preference))
    builder.add_node("tools", ToolNode(refine.REFINER_TOOLS))

    builder.add_conditional_edges(START, route_entry, ["refine_agent", "load_memory"])
    builder.add_conditional_edges("refine_agent", route_after_refine, ["tools", END])
    builder.add_conditional_edges("tools", route_after_tools, ["confirm_preference", "refine_agent"])
    builder.add_conditional_edges("confirm_preference", route_after_tools, ["confirm_preference", "refine_agent"])
    builder.add_conditional_edges("load_memory", route_after_memory, ["ingest_cv", "index_cv", "analyze_job"])
    builder.add_edge("ingest_cv", "index_cv")
    builder.add_edge("index_cv", "plan")
    builder.add_edge("analyze_job", "plan")
    builder.add_edge("plan", "retrieve")
    builder.add_conditional_edges("retrieve", route_after_retrieve, ["retrieve", "generate"])
    builder.add_conditional_edges("generate", route_after_generate, ["validate", "human_escalation"])
    builder.add_conditional_edges("validate", route_after_validate,
                                  ["human_review", "render_and_save", "generate", "judge", "human_escalation"])
    builder.add_conditional_edges("judge", route_by_verdict,
                                  ["human_review", "render_and_save", "generate", "human_escalation"])
    for name in ("human_review", "human_escalation"):
        builder.add_conditional_edges(name, route_after_human, ["render_and_save", "generate", END])
    builder.add_edge("render_and_save", END)
    return builder.compile(checkpointer=checkpointer, store=store)


def graph_mermaid() -> str:
    # Al dibujar, LangGraph agrega aristas hacia el nodo diferido `plan` que no existen en la
    # definición del grafo ni en ninguna ejecución (la conversación nunca vuelve a planificar)
    mermaid = build_graph().get_graph().draw_mermaid()
    return "\n".join(line for line in mermaid.splitlines()
                     if line.strip() not in ("refine_agent -.-> plan;", "confirm_preference -.-> plan;"))
