"""
Grafo cíclico de CVertidor (LangGraph).

Convierte el pipeline lineal del notebook en un grafo con retroalimentación:

        START
          │
          ▼
   ┌──────────────┐     ┌──────────────┐
   │  generate    │────▶│  evaluate     │
   │ (Gemini)     │     │ (OpenRouter   │
   └──────────────┘     │  free)        │
                          └──────┬───────┘
                                 │
                  SÍ ────────────┼─────────────── NO (y aún quedan ciclos)
                  │                              │
                  ▼                              ▼
            ┌──────────┐               ┌──────────────┐
            │ save     │               │ correct       │────┐
            │(Supabase)│               │(Gemini)       │    │
            └────┬─────┘               └──────────────┘    │
                 │                                          │
                 ▼            ◀─────────────────────────────┘
               END             (vuelve a evaluate)

El nodo evaluador consume OpenRouter (free) con un prompt estricto SÍ/NO.
Si el CV es FIEL → se guarda en Supabase. Si NO → se corrige y se vuelve a
evaluar, hasta un tope ``max_iterations`` que impide bucles infinitos.
"""

from __future__ import annotations

from typing import Callable, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from cvertidor.config import MAX_ITERATIONS
from cvertidor.database import save_adapted_cv
from cvertidor.llm import generate_structured_cv
from cvertidor.models import Evaluation, StandardCV
from cvertidor.openrouter import evaluate_cv_faithfulness
from cvertidor.prompts import build_adaptation_prompts, build_correction_prompts


# --------------------------------------------------------------------------- #
# Estado del grafo
# --------------------------------------------------------------------------- #
class AgentState(TypedDict, total=False):
    """
    Estado compartido entre nodos del grafo.

    ``total=False`` permite que cada nodo devuelva solo las claves que actualiza
    (retornos parciales), que LangGraph fusiona en el estado general.
    """

    job_description: str            # oferta de trabajo objetivo
    candidate_id: str              # identificador estable del candidato
    candidate_context: str         # chunks recuperados (fuente de datos única)
    current_cv_text: str           # CV actual en forma de JSON canónico
    eval_verdict: str             # "SÍ" | "NO"
    eval_reason: str             # justificación del veredicto (para retroalimentar)
    is_faithful: bool            # bandera de fidelidad derivada del veredicto
    iterations: int             # nº de rondas de corrección realizadas
    max_iterations: int        # tope de correcciones
    save_result: dict          # id/fila del registro guardado en Supabase
    final_cv: StandardCV          # resultado final validado
    status: str               # rastro de la última acción


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def format_docs(docs) -> str:
    """Formatea los chunk recuperados en un bloque de contexto legible."""
    return "\n\n".join(
        f"--- Chunk ({doc.metadata.get('section', 'N/A')}) ---\n{doc.page_content}"
        for doc in docs
    )


# --------------------------------------------------------------------------- #
# Nodos
# --------------------------------------------------------------------------- #
def generate_node(state: AgentState) -> dict:
    """Adapta el CV al contexto + oferta (generación inicial con Gemini)."""
    system, human = build_adaptation_prompts(
        context=state["candidate_context"],
        job_description=state["job_description"],
    )
    cv: StandardCV = generate_structured_cv(system, human)
    return {
        "current_cv_text": cv.model_dump_json(indent=2),
        "status": "generate: CV inicial generado",
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluador free de OpenRouter: ¿el CV adaptado es fiel al contexto?"""
    evaluation: Evaluation = evaluate_cv_faithfulness(
        context=state["candidate_context"],
        job_description=state["job_description"],
        cv_text=state["current_cv_text"],
    )
    verdict = "SÍ" if evaluation.is_faithful else "NO"
    return {
        "is_faithful": evaluation.is_faithful,
        "eval_verdict": verdict,
        "eval_reason": evaluation.reason,
        "status": (
            f"evaluate: {verdict} — {evaluation.reason[:120]}"
        ),
    }


def correct_node(state: AgentState) -> dict:
    """Corrige el CV anterior guiándose del veredicto del evaluador."""
    system, human = build_correction_prompts(
        context=state["candidate_context"],
        job_description=state["job_description"],
        current_cv=state["current_cv_text"],
        feedback=state.get("eval_reason", ""),
    )
    cv: StandardCV = generate_structured_cv(system, human)
    return {
        "current_cv_text": cv.model_dump_json(indent=2),
        "iterations": state.get("iterations", 0) + 1,
        "status": "correct: CV corregido y reenviado a evaluación",
    }


def save_node(state: AgentState) -> dict:
    """Persiste el CV final en Supabase y finaliza el grafo."""
    cv = StandardCV.model_validate_json(state["current_cv_text"])
    result = save_adapted_cv(
        candidate_id=state["candidate_id"],
        job_description=state["job_description"],
        cv=cv,
        iterations_used=state.get("iterations", 0),
    )
    return {
        "final_cv": cv,
        "save_result": result,
        "status": f"save: registrado en Supabase (id={result['id']})",
    }


# --------------------------------------------------------------------------- #
# Enrutamiento condicional (el "decisor" del grafo cíclico)
# --------------------------------------------------------------------------- #
def route_after_eval(state: AgentState) -> Literal["save", "fix"]:
    """
    Decisor del nodo evaluador.

    - Si el CV es FIEL → ``save`` (guardar y terminar).
    - Si NO es fiel pero aún quedan ciclos → ``fix`` (corregir y volver a evaluar).
    - Si NO es fiel y ya alcanzamos el tope → ``save`` de todos modos (evita
      bucles infinitos y deja el mejor esfuerzo registrado).
    """
    if state.get("is_faithful", False):
        return "save"
    if state.get("iterations", 0) >= state.get("max_iterations", MAX_ITERATIONS):
        return "save"
    return "fix"


# --------------------------------------------------------------------------- #
# Construcción del grafo
# --------------------------------------------------------------------------- #
def build_graph():
    """Construye y compila el grafo cíclico LangGraph."""
    graph: StateGraph = StateGraph(AgentState)

    graph.add_node("generate", generate_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("correct", correct_node)
    graph.add_node("save", save_node)

    graph.add_edge(START, "generate")
    graph.add_edge("generate", "evaluate")

    # El bucle: evaluate decide entre guardar o corregir; correct vuelve a evaluar.
    graph.add_conditional_edges(
        "evaluate",
        route_after_eval,
        {
            "save": "save",
            "fix": "correct",
        },
    )
    graph.add_edge("correct", "evaluate")
    graph.add_edge("save", END)

    return graph.compile()


def run_agent(initial_state: AgentState):
    """
    Ejecuta el grafo con un estado inicial y devuelve el estado final.

    ``initial_state`` debe contener al menos:
    - ``job_description``
    - ``candidate_id``
    - ``candidate_context``
    - ``max_iterations``
    """
    state = {**initial_state, "iterations": 0, "is_faithful": False}
    graph = build_graph()
    final = graph.invoke(state)
    return final
