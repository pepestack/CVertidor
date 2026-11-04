"""
Clientes de IA de CVertidor.

Encapsula:
- ``parse_raw_cv``: conversión de texto libre a ``StandardCV`` con Gemini
  (usan la librería ``google-genai`` con ``response_schema``).
- ``get_structured_generator``: un ``ChatGoogleGenerativeAI`` con salida
  estructurada, listo para alimentar los nodos del grafo LangGraph.
"""

from __future__ import annotations

import json
import re
from typing import Annotated, Sequence

from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import Runnable

from cvertidor.config import (
    EMBEDDING_DIM,
    GEMINI_EMBED_MODEL,
    GEMINI_GEN_MODEL,
    GEMINI_PARSE_MODEL,
    GOOGLE_API_KEY,
)
from cvertidor.models import StandardCV
from cvertidor.prompts import build_parsing_prompt


def _get_google_client() -> genai.Client:
    """Crea un cliente de Google AI Studio con la API key de entorno."""
    from cvertidor.config import GOOGLE_API_KEY

    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY no está configurada en el entorno.")
    return genai.Client(api_key=GOOGLE_API_KEY)


def parse_raw_cv(raw_cv_text: str, model: str = GEMINI_PARSE_MODEL) -> StandardCV:
    """
    Convertir texto libre de un CV a un objeto ``StandardCV`` validado.

    Usa ``google-genai`` con ``response_schema`` para forzar salida estructurada
    y valida el resultado con Pydantic.
    """
    client = _get_google_client()
    system, human = build_parsing_prompt(raw_cv_text)

    response = client.models.generate_content(
        model=model,
        contents=[
            {"role": "user", "parts": [{"text": f"{system}\n\n{human}"}]},
        ],
        config={
            "response_mime_type": "application/json",
            "response_schema": StandardCV,
            "temperature": 0.0,
        },
    )

    text = (response.text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?|```\s*$", "", text, flags=re.MULTILINE).strip()

    return StandardCV.model_validate(json.loads(text))


def get_embeddings() -> "GoogleGenerativeAIEmbeddings":
    """Cliente de embeddings de Google AI Studio para pgvector (768 dims por defecto)."""
    return GoogleGenerativeAIEmbeddings(
        model=GEMINI_EMBED_MODEL,
        output_dimensionality=EMBEDDING_DIM,
        google_api_key=GOOGLE_API_KEY,
    )


def get_structured_generator(
    model: str = GEMINI_GEN_MODEL, temperature: float = 0.1
) -> Runnable[Sequence[Annotated[str, str]] | str, StandardCV]:
    """
    Llama a ``ChatGoogleGenerativeAI`` con ``with_structured_output``, listo para
    invocarse en un nodo LangGraph.

    El tipo de retorno indica que acepta una secuencia de mensajes o un string y
    devuelve un ``StandardCV`` validado.
    """
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=GOOGLE_API_KEY,
    )
    return llm.with_structured_output(StandardCV)


def generate_structured_cv(system: str, human: str, model: str = GEMINI_GEN_MODEL, temperature: float = 0.1) -> StandardCV:
    """Convenciente para nodos que ya tienen (system, human) construidos."""
    llm = get_structured_generator(model=model, temperature=temperature)
    return llm.invoke([("system", system), ("human", human)])
