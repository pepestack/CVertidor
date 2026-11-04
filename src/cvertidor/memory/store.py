"""Memoria de largo plazo en Postgres (PostgresStore), compartida entre hilos y ejecuciones.

Espacios de nombres:
    ("cvs",)                     hash del CV -> candidate_id
    ("candidates", candidate_id) "profile": CV estructurado y su hash
    ("jobs",)                    hash de la oferta -> requisitos extraídos
    ("preferences", candidate_id) preferencias recordadas del usuario
    ("generations", candidate_id) CV generados, con búsqueda semántica por oferta

Todas las lecturas por candidato usan el candidate_id del estado del grafo, nunca uno
indicado por un modelo: así un CV o una oferta maliciosa no alcanza datos de otro candidato.
"""

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from langgraph.store.base import BaseStore
from langgraph.store.postgres import PostgresStore

from cvertidor.llm.embeddings import EmbeddingsInfo
from cvertidor.schemas.cv import StandardCV
from cvertidor.schemas.job import JobRequirements

# Dimensión fija de la tabla de vectores del store (la de gemini-embedding-2 configurada).
# Con otro tamaño (MiniLM, otros modelos) la memoria funciona, pero sin búsqueda semántica
STORE_VECTOR_DIMS = 768
SEMANTIC_FIELD = "text"
MAX_INDEXED_CHARS = 4000


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def semantic_index(info: EmbeddingsInfo) -> dict[str, Any] | None:
    if info.dims != STORE_VECTOR_DIMS:
        return None
    return {"dims": STORE_VECTOR_DIMS, "embed": info.embeddings, "fields": [SEMANTIC_FIELD]}


@contextmanager
def open_store(db_url: str, info: EmbeddingsInfo) -> Iterator[PostgresStore]:
    with PostgresStore.from_conn_string(db_url, index=semantic_index(info)) as store:
        store.setup()  # Idempotente: crea las tablas solo si faltan
        yield store


def load_profile(store: BaseStore, cv_hash: str) -> tuple[str, StandardCV] | None:
    """CV estructurado guardado para este texto exacto; None si es nuevo o fue editado"""
    entry = store.get(("cvs",), cv_hash)
    if entry is None:
        return None
    candidate_id = entry.value["candidate_id"]
    profile = store.get(("candidates", candidate_id), "profile")
    # El mismo candidato pudo subir después otra versión de su CV
    if profile is None or profile.value["cv_hash"] != cv_hash:
        return None
    return candidate_id, StandardCV.model_validate(profile.value["parsed_cv"])


def save_profile(store: BaseStore, candidate_id: str, cv_hash: str, parsed_cv: StandardCV) -> None:
    store.put(("cvs",), cv_hash, {"candidate_id": candidate_id}, index=False)
    store.put(("candidates", candidate_id), "profile",
              {"cv_hash": cv_hash, "parsed_cv": parsed_cv.model_dump(), "updated_at": _now()}, index=False)


def load_job(store: BaseStore, job_hash: str) -> JobRequirements | None:
    entry = store.get(("jobs",), job_hash)
    return JobRequirements.model_validate(entry.value["requirements"]) if entry else None


def save_job(store: BaseStore, job_hash: str, requirements: JobRequirements) -> None:
    store.put(("jobs",), job_hash, {"requirements": requirements.model_dump(), "updated_at": _now()}, index=False)


def load_preferences(store: BaseStore, candidate_id: str) -> list[str]:
    return [item.value[SEMANTIC_FIELD] for item in store.search(("preferences", candidate_id), limit=20)]


def load_preferred_language(store: BaseStore, candidate_id: str) -> str | None:
    """Idioma de salida que la persona pidió recordar; manda sobre el idioma de la oferta"""
    for item in store.search(("preferences", candidate_id), limit=20):
        if item.value.get("output_language"):
            return item.value["output_language"]
    return None


def search_preferences(store: BaseStore, candidate_id: str, query: str = "") -> list[str]:
    """Preferencias del candidato; con `query`, las semánticamente más cercanas primero"""
    items = store.search(("preferences", candidate_id), query=query or None, limit=10)
    return [item.value[SEMANTIC_FIELD] for item in items]


def save_preference(store: BaseStore, candidate_id: str, text: str, output_language: str | None = None) -> None:
    """Idempotente: la misma preferencia actualiza el mismo registro"""
    store.put(("preferences", candidate_id), text_hash(text)[:16],
              {SEMANTIC_FIELD: text, "output_language": output_language, "updated_at": _now()})


def find_preference(store: BaseStore, candidate_id: str, text: str) -> tuple[str, str] | None:
    """(clave, texto) de la preferencia guardada: la exacta o, si no hay, la más parecida.
    La persona ve el texto encontrado antes de borrarla, así que un parecido equivocado se detecta"""
    namespace = ("preferences", candidate_id)
    wanted = text.strip().lower()
    for item in store.search(namespace, limit=20):
        if item.value[SEMANTIC_FIELD].strip().lower() == wanted:
            return item.key, item.value[SEMANTIC_FIELD]
    closest = store.search(namespace, query=text, limit=1)
    return (closest[0].key, closest[0].value[SEMANTIC_FIELD]) if closest else None


def forget_preference(store: BaseStore, candidate_id: str, key: str) -> None:
    store.delete(("preferences", candidate_id), key)


def find_similar_generations(store: BaseStore, candidate_id: str, job_text: str, limit: int = 3) -> list[dict]:
    """CV generados antes para este candidato, de la oferta más parecida a la menos parecida"""
    items = store.search(("generations", candidate_id), query=job_text[:MAX_INDEXED_CHARS], limit=limit)
    return [
        {"job_title": item.value.get("job_title"), "created_at": item.value.get("created_at"),
         "score": round(item.score, 3) if item.score is not None else None}
        for item in items
    ]


def save_generation(store: BaseStore, candidate_id: str, job_text: str, markdown: str, record: dict[str, Any]) -> str:
    """Idempotente: la misma oferta y el mismo CV generado actualizan el mismo registro.
    job_text va enmascarado porque se indexa con el modelo de embeddings"""
    key = text_hash(f"{candidate_id}\n{job_text}\n{markdown}")[:16]
    store.put(("generations", candidate_id), key,
              {SEMANTIC_FIELD: job_text[:MAX_INDEXED_CHARS], "created_at": _now(), **record})
    return key


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
