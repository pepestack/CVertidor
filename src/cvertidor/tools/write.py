"""Herramientas de escritura del agente de refinamiento. Son las únicas que cambian algo."""

from typing import Literal

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command

from cvertidor.tools.query import Runtime, record_tool

# Autor de los avisos internos en el historial: no son mensajes de la persona
SYSTEM_NOTE = "sistema"


def last_person_message(state: dict) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage) and message.name != SYSTEM_NOTE:
            return message.content
    return ""


def _merge(state: dict, update: dict) -> None:
    for key, value in update.items():
        if key in ("warnings", "metrics"):
            state[key] = [*(state.get(key) or []), *value]
        else:
            state[key] = value


def _run_step(name: str, fn, state: dict, ctx) -> None:
    update = fn(state, ctx)
    ctx.audit.record(name, status="ok", origin="edit_cv", **update.pop("audit", {}))
    _merge(state, update)


@tool
def edit_cv(runtime: Runtime, output_language: Literal["es", "en"] | None = None) -> Command:
    """Aplica al CV adaptado el cambio que la persona pidió en su último mensaje y valida el resultado.
    Úsala para cualquier cambio de contenido. Indica `output_language` solo si la persona pidió cambiar de idioma."""
    from cvertidor.graph import nodes

    ctx = runtime.context
    state = dict(runtime.state)
    # Lo que cuenta como dato confirmado es el texto de la persona, nunca uno escrito por el
    # modelo: de lo contrario podría hacer pasar datos inventados como correcciones
    correction = last_person_message(state)
    corrections = list(state.get("human_corrections") or [])
    if correction and correction not in corrections:
        corrections.append(correction)

    plan = state["plan"]
    if output_language and output_language != plan.output_language:
        plan = plan.model_copy(update={
            "output_language": output_language,
            "translate": output_language != plan.cv_language,
            "instructions": f"{plan.instructions}\n- Idioma de salida actualizado por la persona: "
                            f"{nodes.LANGUAGE_NAMES[output_language]}. Traduce todo el CV a ese idioma.",
        })
    state.update(plan=plan, human_corrections=corrections, generation_attempts=0, verdict=None, metrics=[], warnings=[])

    ctx.on_step("Herramienta: edit_cv (editando y validando el CV)")
    _run_step("generate", nodes.generate, state, ctx)
    _run_step("validate", nodes.validate, state, ctx)
    if state["verdict"].status == "uncertain":
        _run_step("judge", nodes.judge, state, ctx)

    verdict = state["verdict"]
    detail = f" Problemas: {'; '.join(verdict.issues)}" if verdict.issues else ""
    ctx.audit.record("tool:edit_cv", status="ok", verdict=verdict.status, corrections=len(corrections))
    return Command(update={
        "plan": plan,
        "human_corrections": corrections,
        "human_decision": None,
        "draft_cv": state["draft_cv"],
        "context": state["context"],
        "verdict": verdict,
        "metrics": state["metrics"],
        "warnings": state["warnings"],
        "messages": [ToolMessage(
            f"CV editado, sin guardar todavía. Validación: {verdict.status}.{detail}",
            tool_call_id=runtime.tool_call_id,
        )],
    })


@tool
def save_cv_files(runtime: Runtime) -> Command | str:
    """Guarda el CV adaptado actual en Markdown y HTML. Usar solo si la persona pidió guardarlo."""
    from cvertidor.graph import nodes

    state, ctx = runtime.state, runtime.context
    verdict = state["verdict"]
    # Un CV rechazado solo se guarda si una persona lo aceptó, en la revisión del flujo principal
    if verdict.status == "rejected" and state.get("human_decision") != "aceptar":
        record_tool(runtime, "save_cv_files", saved=False)
        return f"No se guardó: la validación rechazó el CV. {'; '.join(verdict.issues)}"

    update = nodes.render_and_save(state, ctx)
    record_tool(runtime, "save_cv_files", saved=True, **update.pop("audit", {}))
    # Los nombres de archivo llevan el nombre del candidato: el modelo no los ve
    update["messages"] = [ToolMessage("CV guardado en Markdown y HTML.", tool_call_id=runtime.tool_call_id)]
    return Command(update=update)


@tool
def remember_preference(preference: str, runtime: Runtime, output_language: Literal["es", "en"] | None = None) -> Command:
    """Propone guardar una preferencia para futuros CV (ej. "omitir intereses"). Usar solo si la persona
    pidió recordarla. La persona la confirma antes de guardarla. Indica `output_language` si es sobre el idioma."""
    record_tool(runtime, "remember_preference", output_language=output_language)
    return Command(update={
        "preference_queue": [{"action": "remember", "text": preference, "output_language": output_language}],
        "messages": [ToolMessage("Preferencia enviada a la persona para su confirmación.",
                                 tool_call_id=runtime.tool_call_id)],
    })


@tool
def forget_preference(preference: str, runtime: Runtime) -> Command | str:
    """Propone borrar una preferencia guardada, indicando su texto. Usar solo si la persona pidió
    olvidarla. La persona confirma cuál se borra antes de hacerlo."""
    from cvertidor.graph import nodes

    found = nodes.memory.find_preference(runtime.store, runtime.state["candidate_id"], nodes.scrub_pii(preference))
    record_tool(runtime, "forget_preference", found=bool(found))
    if not found:
        return "No hay preferencias guardadas que coincidan."
    key, text = found
    return Command(update={
        "preference_queue": [{"action": "forget", "key": key, "text": text}],
        "messages": [ToolMessage("Se pidió a la persona confirmar el borrado.", tool_call_id=runtime.tool_call_id)],
    })
