"""Nodos de la conversación sobre un CV ya generado: agente con herramientas y confirmación."""

import time

from langchain_core.messages import BaseMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langgraph.config import get_store
from langgraph.types import interrupt

from cvertidor.graph.nodes import review_markdown
from cvertidor.graph.state import AgentContext, AgentState
from cvertidor.ingestion.pii import scrub_pii
from cvertidor.llm.providers import get_chat_model, get_tool_model, message_text
from cvertidor.memory import store as memory
from cvertidor.pipeline import CANDIDATE_TOKEN, step_metrics
from cvertidor.prompts.refine import REFINE_SYSTEM, SUMMARY_PROMPT
from cvertidor.tools import REFINER_TOOLS
from cvertidor.tools.write import SYSTEM_NOTE

# Tokens de historial por llamada; lo más antiguo se comprime en un resumen. El CV actual va
# siempre aparte, en el mensaje de sistema, así que nunca se pierde
HISTORY_TOKENS = 2000
# Llamadas al modelo por mensaje de la persona: al superarlas debe responder sin herramientas
MAX_REFINE_STEPS = 6
NO_SUMMARY = "(sin mensajes anteriores)"


def _compact_history(state: AgentState, ctx: AgentContext) -> tuple[list[BaseMessage], dict]:
    """Los últimos mensajes que caben en el presupuesto y, si sobran, el resumen de los demás"""
    messages = state["messages"]
    # start_on="human": un turno nunca queda partido entre una llamada y el resultado de su herramienta
    recent = trim_messages(messages, max_tokens=HISTORY_TOKENS, token_counter=count_tokens_approximately,
                           strategy="last", start_on="human")
    if not recent:
        # El turno actual solo no cabe en el presupuesto: se conserva completo
        start = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
        recent = messages[start:]
    kept = {m.id for m in recent}
    dropped = [m for m in messages if m.id not in kept]
    if not dropped:
        return recent, {}

    text = "\n".join(f"{m.type}: {message_text(m)}" for m in dropped if message_text(m))
    prompt = SUMMARY_PROMPT.format(summary=state.get("conversation_summary") or NO_SUMMARY, messages=text)
    try:
        summary = message_text(get_chat_model("refiner", settings=ctx.settings).invoke(prompt))
    except Exception as error:
        # Sin resumen esta vez: los mensajes siguen guardados y se intentará en el próximo turno
        ctx.audit.record("summary", status="error", error=type(error).__name__)
        return recent, {}
    return recent, {"conversation_summary": summary, "removed": [RemoveMessage(id=m.id) for m in dropped]}


def refine_agent(state: AgentState, ctx: AgentContext) -> dict:
    start = time.perf_counter()
    recent, compacted = _compact_history(state, ctx)
    steps = state.get("refine_steps", 0) + 1
    if steps <= MAX_REFINE_STEPS:
        model = get_tool_model("refiner", REFINER_TOOLS, settings=ctx.settings)
    else:
        model = get_chat_model("refiner", settings=ctx.settings)

    job = state["job_reqs"]
    system = SystemMessage(REFINE_SYSTEM.format(
        job_title=job.title or "sin título",
        requirements=", ".join(job.must_have) or "sin requisitos",
        summary=compacted.get("conversation_summary") or state.get("conversation_summary") or NO_SUMMARY,
        draft=review_markdown(state["draft_cv"], state["plan"].output_language),
    ))
    response = model.invoke([system, *recent])

    metrics = step_metrics("refine", start, response)
    update = {
        "messages": [*compacted.get("removed", []), response],
        "refine_steps": steps,
        "metrics": [metrics],
        "audit": {
            "step": steps,
            "tool_calls": [c["name"] for c in response.tool_calls],
            "model": metrics.model,
            "input_tokens": metrics.input_tokens,
            "output_tokens": metrics.output_tokens,
            "history_messages": len(recent),
            "summarized": len(compacted.get("removed", [])),
        },
    }
    if compacted:
        update["conversation_summary"] = compacted["conversation_summary"]
    return update


def confirm_preference(state: AgentState, ctx: AgentContext) -> dict:
    """La memoria de largo plazo solo cambia si la persona lo confirma, para guardar o para borrar.
    Es un nodo aparte, y no un interrupt() dentro de la herramienta, porque al retomar LangGraph
    repite el nodo completo: repetiría también las demás herramientas de ese turno, como una
    edición que ya se pagó"""
    handled = state.get("preferences_handled", 0)
    request = state["preference_queue"][handled]
    forgetting = request["action"] == "forget"
    # La que se va a borrar ya está guardada y enmascarada; la nueva la escribió el modelo
    text = request["text"] if forgetting else (
        scrub_pii(request["text"]).replace(state["parsed_cv"].candidate_name, CANDIDATE_TOKEN))
    question = f"¿Olvido esta preferencia?: {text}" if forgetting else f"¿Guardo esta preferencia para futuros CV?: {text}"
    answer = interrupt({
        "kind": "preference",
        "question": question,
        "options": ["olvidar", "conservar"] if forgetting else ["guardar", "omitir"],
        "preview": "",
        "status": "approved",
        "source": "none",
    })

    action = answer["action"]
    preferences = state.get("preferences", [])
    if action == "guardar":
        memory.save_preference(get_store(), state["candidate_id"], text, request.get("output_language"))
        preferences = [*preferences, text]
    elif action == "olvidar":
        memory.forget_preference(get_store(), state["candidate_id"], request["key"])
        preferences = [p for p in preferences if p != text]
    # Redactado como un hecho consumado: escrito como una decisión, el modelo vuelve a ejecutarla
    outcome = {
        "guardar": "ya quedó guardada",
        "olvidar": "ya quedó borrada",
    }.get(action, "se mantiene sin cambios, por decisión de la persona")
    note = f"Aviso del sistema: la preferencia '{text}' {outcome}. No hace falta otra herramienta para esto; informa el resultado."
    return {
        "preferences_handled": handled + 1,
        "preferences": preferences,
        "messages": [HumanMessage(note, name=SYSTEM_NOTE)],
        "audit": {"decision": action, "request": request["action"], "output_language": request.get("output_language")},
    }
