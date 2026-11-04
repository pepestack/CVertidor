"""
Capa de persistencia de CVertidor.

Cubre:
- Conexión a Supabase (Postgres) vía ``psycopg``.
- Activación de la extensión ``vector`` (pgvector).
- Embeddings de Google AI Studio.
- Recuperación MMR restringida por ``candidate_id``.
- Guardado del CV final adaptado en una tabla JSONB de resultados.

Toda la infra de RAG vive aquí; ``graph.py`` solo orquesta los nodos y
delega los efectos con estado aquí.
"""

from __future__ import annotations

from cvertidor.config import (
    EMBEDDING_DIM,
    GEMINI_EMBED_MODEL,
    GOOGLE_API_KEY,
    RETRIEVER_FETCH_K,
    RETRIEVER_K,
    RETRIEVER_LAMBDA,
    RESULTS_TABLE,
    SUPABASE_URL,
    VECTOR_COLLECTION,
)
from cvertidor.models import StandardCV


# --------------------------------------------------------------------------- #
# Conexión y extensión vector
# --------------------------------------------------------------------------- #
def get_connection():
    """Devuelve una conexión psycopg a Supabase (autogestionada por el llamador)."""
    import psycopg

    if not SUPABASE_URL:
        raise RuntimeError("SUPABASE_URL no está configurada en el entorno.")
    return psycopg.connect(SUPABASE_URL)


def _to_sqlalchemy_url(dsn: str) -> str:
    """Convierte un DSN ``postgresql://...`` al driver de psycopg3 para SQLAlchemy."""
    if not dsn:
        raise RuntimeError("SUPABASE_URL no está configurada en el entorno.")
    if not dsn.startswith("postgresql://"):
        return dsn
    return dsn.replace("postgresql://", "postgresql+psycopg://", 1)


def init_vector_extension() -> None:
    """
    Activa la extensión ``vector`` en la base de datos de Supabase y verifica
    que ya quedó habilitada.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            print(f"Versión Postgres: {cur.fetchone()}")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT '[1,2,3]'::vector;")
            print(f"Prueba vector: {cur.fetchone()}")
    print("✅ Extensión vector activada y verificada.")


# --------------------------------------------------------------------------- #
# Ingesta y recuperación
# --------------------------------------------------------------------------- #
def build_vector_store(documents, candidate_id: str, collection_name: str = VECTOR_COLLECTION) -> "object":
    """
    Crea (o reemplaza) la colección de embeddings de pgvector para un candidato.

    - Usa embeddings de Google AI Studio.
    - Usa JSONB para conservar los metadatos sin cast.
    - Elimina la colección previa para permitir re-ingesta idempotente.
    """
    from langchain_core.documents import Document

    from cvertidor.llm import get_embeddings
    from langchain_postgres import PGVector
    from sqlalchemy import create_engine

    embeddings = get_embeddings()
    connection = _to_sqlalchemy_url(SUPABASE_URL)
    engine = create_engine(connection)

    # Reemplazo de colección existente para idempotencia
    temp_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=engine,
        use_jsonb=True,
    )
    try:
        temp_store.delete_collection()
    except Exception:
        # Si la colección aún no existe, ignorar y seguir con la creación.
        pass

    return PGVector.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        connection=engine,
        use_jsonb=True,
    )


def create_retriever(
    candidate_id: str,
    collection_name: str = VECTOR_COLLECTION,
    k: int = RETRIEVER_K,
    fetch_k: int = RETRIEVER_FETCH_K,
    lambda_mult: float = RETRIEVER_LAMBDA,
):
    """
    Devuelve un retriever MMR con filtro por ``candidate_id``.

    El MMR diversifica los chunk recuperados y evita la redundancia típica de
    búsqueda densa en RAG.
    """
    from cvertidor.llm import get_embeddings
    from langchain_postgres import PGVector
    from sqlalchemy import create_engine

    embeddings = get_embeddings()
    connection = _to_sqlalchemy_url(SUPABASE_URL)
    engine = create_engine(connection)

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=engine,
        use_jsonb=True,
    )
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult,
            "filter": {"candidate_id": candidate_id},
        },
    )


# --------------------------------------------------------------------------- #
# Resultados duraderos del pipeline
# --------------------------------------------------------------------------- #
def _ensure_results_table() -> None:
    """Crea la tabla de resultados si aún no existe."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {RESULTS_TABLE} (
                    id SERIAL PRIMARY KEY,
                    candidate_id TEXT NOT NULL,
                    job_description TEXT NOT NULL,
                    adapted_cv JSONB NOT NULL,
                    iterations_used INT NOT NULL DEFAULT 0,
                    created_at TIMESTAMPTZ DEFAULT now()
                );
                """
            )


def save_adapted_cv(
    candidate_id: str,
    job_description: str,
    cv: StandardCV,
    iterations_used: int = 0,
) -> dict:
    """
    Persiste el resultado final del pipeline en la tabla de resultados.

    Devuelve el id del registro insertado para trazabilidad.
    """
    import psycopg

    _ensure_results_table()
    # psycopg3: usar el adaptador ``Json`` para casting dict/str a JSONB.
    try:
        from psycopg.types.json import Json
    except Exception:  # pragma: no cover - compatibilidad de import
        import psycopg

        Json = getattr(psycopg, "Json", None)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {RESULTS_TABLE}
                (candidate_id, job_description, adapted_cv, iterations_used)
                VALUES (%s, %s, %s, %s)
                RETURNING id, created_at
                """,
                (
                    candidate_id,
                    job_description,
                    Json(cv.model_dump()),
                    iterations_used,
                ),
            )
            row = cur.fetchone()
        conn.commit()

    return {"id": row[0], "created_at": str(row[1])}
