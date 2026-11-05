"""Verificación de compatibilidad de los modelos configurados (`cvertidor check-models`).

Hace una llamada mínima a cada modelo de cada rol (principal y fallbacks) para detectar
antes de una ejecución real si un modelo no existe, no está disponible o no soporta
lo que su rol necesita (salida estructurada o herramientas).
"""

import time
from dataclasses import dataclass
from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel

from cvertidor.config import ROLES, Settings, get_settings
from cvertidor.llm.capabilities import openrouter_capabilities
from cvertidor.llm.embeddings import build_embeddings
from cvertidor.llm.providers import ModelSpec, build_chat_model, resolve_structured_method, role_specs

# Roles que necesitan herramientas; el resto necesita salida estructurada
TOOL_ROLES = {"refiner"}


class _Ping(BaseModel):
    estado: Literal["ok"]
    palabra: str


@tool
def sumar(a: int, b: int) -> int:
    """Suma dos números enteros."""
    return a + b


@dataclass
class CheckResult:
    role: str
    model: str
    capability: str
    ok: bool
    detail: str
    seconds: float
    is_fallback: bool = False


def error_status(error: BaseException) -> int | None:
    """Código HTTP de un error de API: OpenAI/OpenRouter usan status_code y google-genai
    usa code; LangChain envuelve el error de Google, así que también se revisa la causa"""
    for candidate in (error, error.__cause__, error.__context__):
        if candidate is None:
            continue
        for attr in ("status_code", "code"):
            value = getattr(candidate, attr, None)
            if isinstance(value, int):
                return value
    message = str(error)
    if "RESOURCE_EXHAUSTED" in message:
        return 429
    if "UNAVAILABLE" in message:
        return 503
    return None


def describe_error(error: Exception) -> str:
    """Resume un error de API en un mensaje legible"""
    status = error_status(error)
    known = {
        401: "clave de API inválida (401)",
        402: "sin créditos para este modelo (402)",
        404: "modelo no encontrado (404)",
        429: "límite de uso alcanzado (429), reintente más tarde",
        503: "modelo saturado por el proveedor (503), reintente más tarde",
    }
    if status in known:
        return known[status]
    message = str(error).replace("\n", " ")
    return f"{type(error).__name__}: {message[:160]}"


def _check_structured(spec: ModelSpec, role: str, method: str | None, settings: Settings) -> str:
    # Sin reintentos: la prueba informa el estado del modelo en este momento
    model = build_chat_model(spec, role, settings, max_retries=0)
    kwargs = {"method": method} if method else {}
    result = model.with_structured_output(_Ping, **kwargs).invoke(
        "Responde únicamente con estado='ok' y palabra='hola'."
    )
    if not isinstance(result, _Ping):
        raise ValueError(f"respuesta sin el esquema esperado: {result!r}"[:160])
    return f"salida estructurada ({method or 'predeterminada'})"


def _check_tools(spec: ModelSpec, role: str, settings: Settings) -> str:
    model = build_chat_model(spec, role, settings, max_retries=0).bind_tools([sumar])
    response = model.invoke("Usa la herramienta sumar para calcular 2 + 3.")
    calls = [c["name"] for c in response.tool_calls]
    if "sumar" not in calls:
        raise ValueError("el modelo no llamó a la herramienta")
    return "llamada a herramientas"


def check_roles(
    roles: list[str] | None = None,
    settings: Settings | None = None,
    capabilities: dict[str, frozenset[str]] | None = None,
) -> list[CheckResult]:
    settings = settings or get_settings()
    roles = roles or list(ROLES)
    if capabilities is None:
        # El catálogo de OpenRouter solo se consulta si algún rol usa ese proveedor
        uses_openrouter = any(s.provider == "openrouter" for r in roles for s in role_specs(r, settings)[0])
        capabilities = openrouter_capabilities() if uses_openrouter else {}
    caps = capabilities
    results: list[CheckResult] = []
    # Mismo modelo y misma prueba en varios roles: se llama una sola vez
    cache: dict[tuple, tuple[bool, str, float]] = {}

    for role in roles:
        specs, override = role_specs(role, settings)
        for index, spec in enumerate(specs):
            if role in TOOL_ROLES:
                key = (str(spec), "tools")
                run = lambda: _check_tools(spec, role, settings)
            else:
                method = resolve_structured_method(spec, override, caps)
                key = (str(spec), "structured", method)
                run = lambda: _check_structured(spec, role, method, settings)

            if key not in cache:
                start = time.perf_counter()
                try:
                    cache[key] = (True, run(), time.perf_counter() - start)
                except Exception as error:
                    cache[key] = (False, describe_error(error), time.perf_counter() - start)

            ok, detail, seconds = cache[key]
            results.append(CheckResult(role, str(spec), "tools" if role in TOOL_ROLES else "structured",
                                       ok, detail, seconds, is_fallback=index > 0))
    return results


def check_embeddings(settings: Settings | None = None) -> CheckResult:
    settings = settings or get_settings()
    spec = ModelSpec.parse(settings.embedding_model)
    start = time.perf_counter()
    try:
        vector = build_embeddings(spec, settings).embed_query("Ingeniero de datos con experiencia en Python")
        return CheckResult("embeddings", str(spec), "embeddings", True,
                           f"{len(vector)} dimensiones", time.perf_counter() - start)
    except Exception as error:
        return CheckResult("embeddings", str(spec), "embeddings", False,
                           f"{describe_error(error)} (se usará MiniLM local)", time.perf_counter() - start)


def check_decision_model(settings: Settings | None = None) -> CheckResult:
    from cvertidor.llm.decisions import DecisionError, evaluate_draft

    settings = settings or get_settings()
    start = time.perf_counter()
    try:
        result = evaluate_draft("Rol: Data Engineer. Responsabilidades: pipelines con Python.", ["Python"],
                                "Data Engineer con pipelines en Python.", settings)
        return CheckResult("decision", settings.decision_model, "decisions", True,
                           f"{len(result.scores)} conductas evaluadas", time.perf_counter() - start)
    except DecisionError as error:
        return CheckResult("decision", settings.decision_model, "decisions", False,
                           f"{str(error)[:160]} (decidirá el juez)", time.perf_counter() - start)


def format_report(results: list[CheckResult]) -> str:
    lines = []
    for r in results:
        status = "OK   " if r.ok else "FALLA"
        kind = "fallback " if r.is_fallback else "principal"
        lines.append(f"[{status}] {r.role:<11} {kind}  {r.model}\n          {r.detail} ({r.seconds:.1f} s)")
    return "\n".join(lines)


def roles_without_working_model(results: list[CheckResult]) -> list[str]:
    """Roles de chat en que no responde ningún modelo (ni principal ni fallbacks)"""
    by_role: dict[str, bool] = {}
    for r in results:
        # Embeddings y modelo de decisión tienen su propio respaldo
        if r.role not in ("embeddings", "decision"):
            by_role[r.role] = by_role.get(r.role, False) or r.ok
    return [role for role, ok in by_role.items() if not ok]
