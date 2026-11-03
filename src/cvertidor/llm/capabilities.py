"""Consulta de las capacidades de cada modelo publicadas en el catálogo de OpenRouter."""

import logging
from functools import lru_cache

import httpx

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

logger = logging.getLogger(__name__)


@lru_cache
def openrouter_capabilities() -> dict[str, frozenset[str]]:
    """Mapa modelo -> parámetros soportados (ej. 'tools', 'structured_outputs').
    Si el catálogo no responde se devuelve vacío y se usan métodos por defecto."""
    try:
        response = httpx.get(OPENROUTER_MODELS_URL, timeout=20)
        response.raise_for_status()
    except httpx.HTTPError as error:
        logger.warning("No se pudo leer el catálogo de OpenRouter: %s", error)
        return {}

    return {
        model["id"]: frozenset(model.get("supported_parameters") or [])
        for model in response.json().get("data", [])
    }


def pick_structured_method(supported: frozenset[str] | None) -> str:
    """Elige cómo pedir salida estructurada según lo que soporta el modelo:
    - function_calling: se fuerza una herramienta con el esquema como argumentos
    - json_schema: decodificación restringida por el proveedor
    - json_mode: solo garantiza JSON válido, el esquema va en el prompt

    function_calling va primero: con nemotron-3.5-super-vl, json_schema borró la palabra
    "Español" de "Español: Nativo" en todas las corridas, mientras function_calling la
    conservó con la misma velocidad y el mismo resultado en el resto de campos.
    """
    if not supported:
        return "function_calling"
    if "tools" in supported:
        return "function_calling"
    if "structured_outputs" in supported:
        return "json_schema"
    if "response_format" in supported:
        return "json_mode"
    return "function_calling"
