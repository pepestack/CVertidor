"""Modelo de embeddings con respaldo local (MiniLM) si el proveedor no está disponible."""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.embeddings import Embeddings

from cvertidor.config import Settings, get_settings
from cvertidor.llm.providers import OPENROUTER_BASE_URL, ModelConfigError, ModelSpec

MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingsInfo:
    embeddings: Embeddings
    # Identifica la colección en pgvector: vectores de modelos distintos no son comparables
    slug: str
    model: str
    is_fallback: bool = False


def embedding_slug(spec: ModelSpec) -> str:
    """Nombre corto y estable del modelo. 'google' se mantiene por las colecciones del notebook"""
    if spec.provider == "google":
        return "google"
    name = spec.model.split("/")[-1].removesuffix(":free")
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def build_embeddings(spec: ModelSpec, settings: Settings) -> Embeddings:
    if spec.provider == "openrouter":
        if not settings.openrouter_api_key:
            raise ModelConfigError("Falta OPENROUTER_API_KEY en las variables de entorno.")
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=spec.model,
            api_key=settings.openrouter_api_key,
            base_url=OPENROUTER_BASE_URL,
            # El tokenizador de OpenAI no aplica a modelos de otros proveedores
            check_embedding_ctx_length=False,
            max_retries=settings.llm_max_retries,
            timeout=settings.llm_timeout,
        )

    if spec.provider == "google":
        if not settings.google_api_key:
            raise ModelConfigError("Falta GOOGLE_API_KEY (o API_KEY) en las variables de entorno.")
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=spec.model,
            output_dimensionality=768,
            google_api_key=settings.google_api_key,
        )

    raise ModelConfigError(f"Proveedor de embeddings desconocido: '{spec.provider}'")


def minilm_embeddings() -> Embeddings:
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=MINILM_MODEL)


def get_embeddings(
    settings: Settings | None = None,
    fallback_factory: Callable[[], Embeddings] = minilm_embeddings,
) -> EmbeddingsInfo:
    """Embeddings del proveedor configurado; si fallan (sin red, límite de uso, sin clave)
    se usa MiniLM local, igual que el fallback del notebook"""
    settings = settings or get_settings()
    spec = ModelSpec.parse(settings.embedding_model)

    try:
        embeddings = build_embeddings(spec, settings)
        embeddings.embed_query("test")
        return EmbeddingsInfo(embeddings, embedding_slug(spec), str(spec))
    except Exception as error:
        logger.warning(
            "Embeddings de %s no disponibles (%s: %s). Se usará MiniLM local.",
            spec, type(error).__name__, error,
        )
        return EmbeddingsInfo(fallback_factory(), "minilm", MINILM_MODEL, is_fallback=True)
