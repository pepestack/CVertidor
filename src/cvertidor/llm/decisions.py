"""Cliente de la Decisions API de OpenRouter (modelos que devuelven probabilidades, no texto).

La API está en alfa: todo lo que depende de su formato queda en este módulo. El estado va
como conversación porque Respan solo acepta ese formato (input + output del asistente);
los demás modelos aceptan cualquier objeto, así que cambiar a Jev es solo cambiar DECISION_MODEL.
"""

from dataclasses import dataclass

import httpx

from cvertidor.config import Settings

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
DECISION_TIMEOUT = 60.0

# Conductas indebidas que se evalúan, con su instrucción al modelo y el texto para corregirlas
BEHAVIORS = {
    "inventa_datos": {
        "instructions": (
            "Does the assistant output claim a tool, skill, employer, job title, certification, metric or "
            "achievement that is absent from the candidate context in the input? Rewording, summarizing or "
            "translating a fact the candidate context contains is not a new claim."
        ),
        "feedback": "Elimina todo dato, herramienta, cifra o logro que no aparezca explícitamente en el contexto del candidato.",
    },
    "omite_requisitos": {
        "instructions": (
            "Does the candidate context in the input show experience with a required item of the job "
            "that the assistant output leaves out?"
        ),
        "feedback": "Incluye la experiencia del contexto del candidato que responde a los requisitos obligatorios de la oferta.",
    },
}


class DecisionError(RuntimeError):
    """El modelo de decisión no respondió o su respuesta no tiene el formato esperado"""


@dataclass
class DecisionResult:
    scores: dict[str, float]
    model: str | None = None
    input_tokens: int | None = None


def evaluate_draft(
    context: str,
    requirements: list[str],
    draft_markdown: str,
    settings: Settings,
    transport: httpx.BaseTransport | None = None,
) -> DecisionResult:
    """Probabilidad de cada conducta indebida. Todo el texto debe llegar enmascarado"""
    if not settings.openrouter_api_key:
        raise DecisionError("Falta OPENROUTER_API_KEY para usar el modelo de decisión.")

    prompt = (
        f"Adapt the CV to the job.\n\nCandidate context:\n{context}"
        f"\n\nRequired items of the job: {', '.join(requirements) or 'none'}"
    )
    body = {
        "model": settings.decision_model,
        "state": {"input": [{"role": "user", "content": prompt}], "output": {"role": "assistant", "content": draft_markdown}},
        "questions": {name: {"type": "noul", "instructions": b["instructions"]} for name, b in BEHAVIORS.items()},
    }
    headers = {"Authorization": f"Bearer {settings.openrouter_api_key.get_secret_value()}", "X-Title": "CVertidor"}
    try:
        with httpx.Client(transport=transport, timeout=DECISION_TIMEOUT) as client:
            response = client.post(DECISIONS_URL, json=body, headers=headers)
        response.raise_for_status()
        data = response.json()
    except (httpx.HTTPError, ValueError) as error:
        raise DecisionError(f"{type(error).__name__}: {str(error)[:200]}") from error

    scores = {}
    for name in BEHAVIORS:
        answer = (data.get("answers") or {}).get(name) or {}
        if answer.get("type") != "noul" or not isinstance(answer.get("noul"), (int, float)):
            raise DecisionError(f"Respuesta inesperada para '{name}': {str(answer)[:200]}")
        scores[name] = round(float(answer["noul"]), 4)
    return DecisionResult(scores, data.get("model"), (data.get("usage") or {}).get("input_tokens"))
