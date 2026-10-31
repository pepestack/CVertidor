"""Fábrica de modelos de chat por rol.

Es el único módulo que conoce a los proveedores (OpenRouter, Google). El resto del
proyecto usa solo la interfaz común de LangChain (invoke, with_structured_output, bind_tools).
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable

from cvertidor.config import Settings, get_settings
from cvertidor.llm.capabilities import openrouter_capabilities, pick_structured_method

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
PROVIDERS = ("openrouter", "google")

# Parámetros de generación por rol (temperatura baja donde importa la fidelidad a los datos)
ROLE_PARAMS: dict[str, dict[str, Any]] = {
    "parser": {"temperature": 0.0, "max_tokens": 4096},
    "job": {"temperature": 0.0, "max_tokens": 2048},
    "generator": {"temperature": 0.1, "top_p": 0.9, "max_tokens": 4096},
    "refiner": {"temperature": 0.2, "max_tokens": 4096},
    "escalation": {"temperature": 0.0, "max_tokens": 2048},
}


class ModelConfigError(ValueError):
    """Configuración de modelos inválida o incompleta (ej. falta una clave de API)"""


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str

    @classmethod
    def parse(cls, value: str) -> "ModelSpec":
        """'openrouter:nvidia/x:free' -> (openrouter, nvidia/x:free). Sin prefijo -> openrouter"""
        value = value.strip()
        prefix, sep, rest = value.partition(":")
        if sep and prefix in PROVIDERS:
            return cls(prefix, rest)
        return cls("openrouter", value)

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}"


def build_chat_model(
    spec: ModelSpec,
    role: str,
    settings: Settings | None = None,
    max_retries: int | None = None,
) -> BaseChatModel:
    """Instancia el modelo de un proveedor traduciendo los parámetros comunes"""
    settings = settings or get_settings()
    params = ROLE_PARAMS[role]
    retries = settings.llm_max_retries if max_retries is None else max_retries

    if spec.provider == "openrouter":
        if not settings.openrouter_api_key:
            raise ModelConfigError("Falta OPENROUTER_API_KEY en las variables de entorno.")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=spec.model,
            api_key=settings.openrouter_api_key,
            base_url=OPENROUTER_BASE_URL,
            temperature=params["temperature"],
            top_p=params.get("top_p"),
            max_tokens=params["max_tokens"],
            max_retries=retries,
            timeout=settings.llm_timeout,
            # Identifica la aplicación en las estadísticas de OpenRouter
            default_headers={"X-Title": "CVertidor"},
        )

    if spec.provider == "google":
        if not settings.google_api_key:
            raise ModelConfigError("Falta GOOGLE_API_KEY (o API_KEY) en las variables de entorno.")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=spec.model,
            google_api_key=settings.google_api_key,
            temperature=params["temperature"],
            top_p=params.get("top_p"),
            max_output_tokens=params["max_tokens"],
            max_retries=retries,
            timeout=settings.llm_timeout,
        )

    raise ModelConfigError(f"Proveedor desconocido: '{spec.provider}'. Use: {', '.join(PROVIDERS)}")


def resolve_structured_method(
    spec: ModelSpec,
    override: str | None = None,
    capabilities: dict[str, frozenset[str]] | None = None,
) -> str | None:
    """Método de salida estructurada para un modelo. None = el predeterminado del proveedor"""
    if override:
        return override
    if spec.provider != "openrouter":
        return None
    caps = capabilities if capabilities is not None else openrouter_capabilities()
    return pick_structured_method(caps.get(spec.model))


def role_specs(role: str, settings: Settings | None = None) -> tuple[list[ModelSpec], str | None]:
    settings = settings or get_settings()
    models, method = settings.role_models(role)
    return [ModelSpec.parse(m) for m in models], method


def with_model_fallbacks(runnables: Sequence[Runnable]) -> Runnable:
    """Encadena el modelo principal con sus respaldos: si uno falla (429, caída, error
    de formato) se intenta el siguiente. Cada uno ya trae su propio método de salida."""
    if not runnables:
        raise ModelConfigError("El rol no tiene modelos configurados.")
    primary, *fallbacks = runnables
    return primary.with_fallbacks(fallbacks) if fallbacks else primary


def _build_chain(
    role: str,
    adapt: Callable[[BaseChatModel, ModelSpec], Runnable],
    settings: Settings | None = None,
) -> Runnable:
    settings = settings or get_settings()
    specs, _ = role_specs(role, settings)
    last = len(specs) - 1
    # Solo el último modelo reintenta: ante un 429 los anteriores ceden de inmediato al respaldo
    return with_model_fallbacks([
        adapt(build_chat_model(spec, role, settings, max_retries=None if i == last else 0), spec)
        for i, spec in enumerate(specs)
    ])


def get_chat_model(role: str, settings: Settings | None = None) -> Runnable:
    """Modelo de texto libre del rol, con sus fallbacks"""
    return _build_chain(role, lambda model, spec: model, settings)


def get_structured_model(
    role: str,
    schema: type,
    *,
    include_raw: bool = False,
    settings: Settings | None = None,
    capabilities: dict[str, frozenset[str]] | None = None,
) -> Runnable:
    """Modelo del rol que responde con una instancia validada de `schema`"""
    settings = settings or get_settings()
    _, override = role_specs(role, settings)

    def adapt(model: BaseChatModel, spec: ModelSpec) -> Runnable:
        method = resolve_structured_method(spec, override, capabilities)
        kwargs = {"include_raw": include_raw}
        if method:
            kwargs["method"] = method
        return model.with_structured_output(schema, **kwargs)

    return _build_chain(role, adapt, settings)


def get_tool_model(role: str, tools: Sequence[Any], settings: Settings | None = None) -> Runnable:
    """Modelo del rol con herramientas vinculadas (para el agente de refinamiento)"""
    return _build_chain(role, lambda model, spec: model.bind_tools(tools), settings)


def message_text(message: BaseMessage | str) -> str:
    """Texto plano de una respuesta: algunos proveedores devuelven bloques de contenido
    (texto, razonamiento, etc.) en vez de un string"""
    content = message if isinstance(message, str) else message.content
    if isinstance(content, str):
        return content.strip()

    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type", "text") == "text" and "text" in block:
            parts.append(block["text"])
    return "\n".join(parts).strip()
