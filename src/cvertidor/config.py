"""
Configuración central del proyecto CVertidor.

Reemplaza el uso de ``google.colab.userdata`` por variables de entorno reales
leídas desde un archivo ``.env`` (ver ``.env.example``). Toda credencial sensible
se mantiene fuera del código fuente y del control de versiones.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

# Carga variables de .env en memoria. ``override=False`` respeta lo que ya esté
# definido en el entorno del sistema (útil en CI/CD o contenedores).
load_dotenv(override=False)


def _get_required(key: str) -> str:
    """Devuelve una variable obligatoria o lanza un error claro si falta."""
    value = os.getenv(key)
    if not value:
        raise RuntimeError(
            f"Variable de entorno obligatoria ausente: {key}. "
            "Cópiala de .env.example a .env y rellena los valores."
        )
    return value


# --------------------------------------------------------------------------- #
# Credenciales
# --------------------------------------------------------------------------- #
GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY")
SUPABASE_URL: str | None = os.getenv("SUPABASE_URL")  # DSN Postgres de Supabase
SUPABASE_KEY: str | None = os.getenv("SUPABASE_KEY")  # Optional (REST/PostgREST)

# OpenRouter (modelo evaluador gratuito)
OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
# Modelo 100% gratuito: sin tokens de pago, contextos pequeños, sin structured output.
OPENROUTER_EVAL_MODEL: str = os.getenv(
    "OPENROUTER_EVAL_MODEL", "meta-llama/llama-3-8b-instruct:free"
)

# --------------------------------------------------------------------------- #
# Modelos y parámetros de Google AI Studio (Gemini)
# --------------------------------------------------------------------------- #
# Ajusta estos nombres reales de modelo según tu plan / disponibilidad.
GEMINI_PARSE_MODEL: str = os.getenv("GEMINI_PARSE_MODEL", "gemini-3.6-flash")
GEMINI_GEN_MODEL: str = os.getenv("GEMINI_GEN_MODEL", "gemini-3.6-flash")
GEMINI_EMBED_MODEL: str = os.getenv("GEMINI_EMBED_MODEL", "models/gemini-embedding-2")
EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "768"))

# --------------------------------------------------------------------------- #
# Pipeline / RAG
# --------------------------------------------------------------------------- #
VECTOR_COLLECTION: str = os.getenv("VECTOR_COLLECTION", "cv_embeddings")
ADAPTED_COLLECTION: str = os.getenv("ADAPTED_COLLECTION", "adapted_cv_embeddings")
RESULTS_TABLE: str = os.getenv("RESULTS_TABLE", "adapted_cv_results")

RETRIEVER_K: int = int(os.getenv("RETRIEVER_K", "6"))
RETRIEVER_FETCH_K: int = int(os.getenv("RETRIEVER_FETCH_K", "20"))
RETRIEVER_LAMBDA: float = float(os.getenv("RETRIEVER_LAMBDA", "0.7"))

# Nº máximo de ciclos de corrección antes de forzar el guardado (impide bucles infinitos).
MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "3"))


def validate() -> None:
    """
    Comprueba que estén disponibles las credenciales mínimas para ejecutar el grafo.
    Llámalo al inicio del programa para fallar rápido con un mensaje accionable.
    """
    missing = [
        key
        for key, value in (
            ("GOOGLE_API_KEY", GOOGLE_API_KEY),
            ("SUPABASE_URL", SUPABASE_URL),
            ("OPENROUTER_API_KEY", OPENROUTER_API_KEY),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Faltan credenciales para ejecutar CVertidor: "
            + ", ".join(missing)
            + ". Edita el archivo .env (ver .env.example)."
        )
