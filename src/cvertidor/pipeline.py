"""Flujo completo del notebook como funciones: extraer el CV, indexarlo, recuperar
lo relevante para la oferta, generar el CV adaptado y guardarlo en Markdown y HTML.

En el batch 4 estas etapas pasan a ser nodos de un grafo de LangGraph.
"""

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from pydantic import BaseModel

from cvertidor.config import Settings, get_settings
from cvertidor.ingestion.pii import MaskResult, mask_pii, unmask
from cvertidor.llm.embeddings import EmbeddingsInfo, get_embeddings
from cvertidor.llm.providers import StructuredOutputError, get_structured_model
from cvertidor.prompts.generation import GENERATION_PROMPT
from cvertidor.prompts.parsing import PARSING_PROMPT
from cvertidor.rag.chunking import create_cv_documents, make_candidate_id
from cvertidor.rag.retrieval import ALWAYS_INCLUDED_SECTIONS, build_retriever, format_docs, similarity_threshold
from cvertidor.rag.vector_store import SyncStats, get_vector_store, sync_candidate_docs
from cvertidor.render import cv_to_html, cv_to_markdown
from cvertidor.schemas.cv import StandardCV

MIN_CV_LENGTH = 50              # Largo minimo de CV
MIN_JOB_DESC_LENGTH = 30        # Largo minimo de oferta de trabajo

# Datos de contacto: se restauran desde los marcadores y deben existir en el CV original
CONTACT_FIELDS = ("email", "phone", "linkedin_url", "portfolio_url")


class InputError(ValueError):
    """Texto de entrada vacío o demasiado corto"""


class GenerationError(RuntimeError):
    """El modelo no devolvió un StandardCV válido"""


@dataclass
class StepMetrics:
    step: str
    seconds: float
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class PipelineResult:
    candidate_id: str
    parsed_cv: StandardCV
    adapted_cv: StandardCV
    markdown: str
    html: str
    output_paths: list[Path]
    retrieved_docs: list[Document]
    sync: SyncStats
    embeddings_model: str
    warnings: list[str] = field(default_factory=list)
    metrics: list[StepMetrics] = field(default_factory=list)


def validate_text(text: str | None, min_length: int, label: str) -> str:
    """Texto limpio; a diferencia del notebook no se reemplaza por datos de ejemplo"""
    cleaned = (text or "").strip()
    if len(cleaned) < min_length:
        raise InputError(f"{label} está vacío o tiene menos de {min_length} caracteres.")
    return cleaned


def _compact(value: str) -> str:
    """Forma comparable de un dato de contacto: sin esquema, espacios ni separadores"""
    value = re.sub(r"(?i)\b(https?://)?(www\.)?", "", value.strip())
    return re.sub(r"[\s()./:-]", "", value).lower()


def unmask_model(model: BaseModel, mapping: dict[str, str]) -> BaseModel:
    """Restaura los marcadores de PII en todos los campos de texto del modelo"""
    def walk(value: Any) -> Any:
        if isinstance(value, str):
            return unmask(value, mapping)
        if isinstance(value, list):
            return [walk(v) for v in value]
        if isinstance(value, dict):
            return {k: walk(v) for k, v in value.items()}
        return value

    return type(model).model_validate(walk(model.model_dump()))


def restore_contact(parsed: StandardCV, masked: MaskResult, original_text: str) -> tuple[StandardCV, list[str]]:
    """Restaura los datos de contacto y descarta los que no aparecen en el CV original
    (los modelos gratuitos tienden a inventar correos, teléfonos o perfiles de LinkedIn)"""
    restored = unmask_model(parsed, masked.mapping)
    source = _compact(original_text)
    removed = []
    for name in CONTACT_FIELDS:
        value = getattr(restored, name)
        if value and _compact(value) not in source:
            removed.append(f"{name}={value!r}")
            setattr(restored, name, None)
    return restored, removed


def _metrics(step: str, start: float, raw: Any = None) -> StepMetrics:
    usage = getattr(raw, "usage_metadata", None) or {}
    model = (getattr(raw, "response_metadata", None) or {}).get("model_name")
    return StepMetrics(step, time.perf_counter() - start, model, usage.get("input_tokens"), usage.get("output_tokens"))


def _invoke_structured(chain, inputs: dict, what: str):
    try:
        result = chain.invoke(inputs)
    except StructuredOutputError as error:
        # Ningún modelo de la cadena (principal ni fallbacks) devolvió una salida válida
        raise GenerationError(f"El modelo no devolvió un {what} válido: {error}") from error
    if result["parsed"] is None:
        raw = result["raw"]
        finish_reason = (raw.response_metadata or {}).get("finish_reason")
        raise GenerationError(
            f"El modelo no devolvió un {what} válido (finish_reason={finish_reason}): {result.get('parsing_error')}"
        )
    return result["parsed"], result["raw"]


def parse_cv(cv_text: str, settings: Settings) -> tuple[StandardCV, list[str], Any]:
    """Estructura el CV original. El modelo solo ve el texto con la PII enmascarada"""
    masked = mask_pii(cv_text)
    chain = PARSING_PROMPT | get_structured_model("parser", StandardCV, include_raw=True, settings=settings)
    parsed, raw = _invoke_structured(chain, {"cv_text": masked.text}, "StandardCV del CV original")
    restored, removed = restore_contact(parsed, masked, cv_text)
    return restored, removed, raw


def generate_adapted_cv(context: str, job_text: str, parsed_cv: StandardCV, settings: Settings) -> tuple[StandardCV, Any]:
    chain = GENERATION_PROMPT | get_structured_model("generator", StandardCV, include_raw=True, settings=settings)
    adapted, raw = _invoke_structured(
        chain, {"context": context, "job_description": mask_pii(job_text).text}, "StandardCV adaptado"
    )
    # Nombre y datos de contacto se copian del CV original, no del modelo
    adapted.candidate_name = parsed_cv.candidate_name
    adapted.location = parsed_cv.location
    for name in CONTACT_FIELDS:
        setattr(adapted, name, getattr(parsed_cv, name))
    return adapted, raw


def ensure_db_available(db_url: str) -> None:
    """Falla rápido si la base no responde, antes de gastar llamadas a modelos (cupo diario)"""
    import psycopg

    with psycopg.connect(db_url, connect_timeout=5) as conn:
        conn.execute("SELECT 1")


def save_outputs(candidate_id: str, markdown: str, html: str, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"cv_{candidate_id}.md", output_dir / f"cv_{candidate_id}.html"]
    paths[0].write_text(markdown, encoding="utf-8")
    paths[1].write_text(html, encoding="utf-8")
    return paths


def run_pipeline(
    cv_text: str,
    job_text: str,
    settings: Settings | None = None,
    embeddings_info: EmbeddingsInfo | None = None,
    on_step: Callable[[str], None] = lambda message: None,
) -> PipelineResult:
    settings = settings or get_settings()
    cv_text = validate_text(cv_text, MIN_CV_LENGTH, "El CV")
    job_text = validate_text(job_text, MIN_JOB_DESC_LENGTH, "La oferta de trabajo")
    db_url = settings.require_db_url()
    ensure_db_available(db_url)
    warnings: list[str] = []
    metrics: list[StepMetrics] = []

    on_step("Extrayendo la estructura del CV (datos personales enmascarados)...")
    start = time.perf_counter()
    parsed_cv, removed, raw = parse_cv(cv_text, settings)
    metrics.append(_metrics("parse", start, raw))
    if removed:
        warnings.append("Se descartaron datos de contacto que no están en el CV original: " + ", ".join(removed))

    candidate_id = make_candidate_id(parsed_cv.candidate_name)
    docs = create_cv_documents(parsed_cv)

    on_step(f"Indexando {len(docs)} fragmentos en pgvector...")
    start = time.perf_counter()
    embeddings_info = embeddings_info or get_embeddings(settings)
    if embeddings_info.is_fallback:
        warnings.append(f"Se usaron embeddings locales ({embeddings_info.model}) porque el proveedor no respondió.")
    vector_store = get_vector_store(embeddings_info.embeddings, embeddings_info.slug, db_url)
    sync = sync_candidate_docs(vector_store, db_url, candidate_id, docs)
    metrics.append(_metrics("index", start))

    on_step("Recuperando la experiencia relevante para la oferta...")
    start = time.perf_counter()
    retriever = build_retriever(
        vector_store, embeddings_info.embeddings, candidate_id, similarity_threshold(embeddings_info.slug)
    )
    retrieved_docs = retriever.invoke(job_text)
    metrics.append(_metrics("retrieve", start))
    if not any(d.metadata.get("section") == "work_history" for d in retrieved_docs):
        warnings.append("Ninguna experiencia laboral superó el filtro de similitud; revise el umbral.")

    always_included = [d for d in docs if d.metadata["section"] in ALWAYS_INCLUDED_SECTIONS]
    context = format_docs(retrieved_docs + always_included)

    on_step("Generando el CV adaptado a la oferta...")
    start = time.perf_counter()
    adapted_cv, raw = generate_adapted_cv(context, job_text, parsed_cv, settings)
    metrics.append(_metrics("generate", start, raw))

    markdown, html = cv_to_markdown(adapted_cv), cv_to_html(adapted_cv)
    paths = save_outputs(candidate_id, markdown, html, settings.output_dir)

    return PipelineResult(
        candidate_id=candidate_id,
        parsed_cv=parsed_cv,
        adapted_cv=adapted_cv,
        markdown=markdown,
        html=html,
        output_paths=paths,
        retrieved_docs=retrieved_docs,
        sync=sync,
        embeddings_model=embeddings_info.model,
        warnings=warnings,
        metrics=metrics,
    )
