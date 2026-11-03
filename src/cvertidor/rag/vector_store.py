"""Almacenamiento de los fragmentos del CV en PostgreSQL con pgvector."""

import re
from dataclasses import dataclass

import psycopg
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector

from cvertidor.rag.chunking import make_doc_id


def sqlalchemy_url(db_url: str) -> str:
    """Inserta el driver de psycopg en la URL si no lo trae (como en Supabase)"""
    return re.sub(r"^postgres(ql)?://", "postgresql+psycopg://", db_url)


def collection_name(embedding_slug: str) -> str:
    """Una colección por modelo de embeddings: vectores de modelos distintos no son comparables"""
    return f"cv_embeddings_{embedding_slug}"


def get_vector_store(embeddings: Embeddings, embedding_slug: str, db_url: str) -> PGVector:
    # Se conecta a la colección (se crea si no existe). No se borra: se conservan los CVs de otros candidatos
    return PGVector(
        embeddings=embeddings,
        collection_name=collection_name(embedding_slug),
        connection=sqlalchemy_url(db_url),
        use_jsonb=True,
    )


def fetch_candidate_ids(db_url: str, collection: str, candidate_id: str) -> set[str]:
    """IDs de los chunks ya guardados para este candidato en la colección"""
    with psycopg.connect(db_url) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT e.id
            FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON e.collection_id = c.uuid
            WHERE c.name = %s AND e.cmetadata->>'candidate_id' = %s
            """,
            (collection, candidate_id),
        )
        return {row[0] for row in cur.fetchall()}


@dataclass
class SyncStats:
    added: int
    removed: int
    unchanged: int


def sync_candidate_docs(
    vector_store: PGVector,
    db_url: str,
    candidate_id: str,
    docs: list[Document],
) -> SyncStats:
    """Sincroniza solo los chunks de este candidato usando IDs deterministas:
    - el dict descarta chunks repetidos (mismo ID) dentro del mismo CV
    - re-ejecutar no duplica nada; un CV editado solo re-embebe lo que cambió"""
    collection = vector_store.collection_name
    new_docs = {make_doc_id(collection, d): d for d in docs}
    existing_ids = fetch_candidate_ids(db_url, collection, candidate_id)

    ids_to_add = [doc_id for doc_id in new_docs if doc_id not in existing_ids]
    stale_ids = existing_ids - new_docs.keys()

    # Primero insertar y luego borrar: si algo falla, el candidato nunca queda sin datos
    if ids_to_add:
        vector_store.add_documents([new_docs[doc_id] for doc_id in ids_to_add], ids=ids_to_add)
    if stale_ids:
        vector_store.delete(ids=list(stale_ids), collection_only=True)

    return SyncStats(len(ids_to_add), len(stale_ids), len(new_docs) - len(ids_to_add))
