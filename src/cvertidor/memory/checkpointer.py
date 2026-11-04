"""Memoria de corto plazo: el estado del grafo se guarda en Postgres por cada thread_id."""

from collections.abc import Iterator
from contextlib import contextmanager

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

# Clases propias que el estado del grafo guarda en los checkpoints. LangGraph solo
# reconstruye al leer las clases autorizadas explícitamente
CHECKPOINT_TYPES = [
    ("cvertidor.schemas.cv", "StandardCV"),
    ("cvertidor.schemas.cv", "Education"),
    ("cvertidor.schemas.cv", "WorkExperience"),
    ("cvertidor.schemas.cv", "Certification"),
    ("cvertidor.schemas.job", "JobRequirements"),
    ("cvertidor.graph.planning", "Plan"),
    ("cvertidor.pipeline", "StepMetrics"),
    ("cvertidor.rag.vector_store", "SyncStats"),
]


@contextmanager
def open_checkpointer(db_url: str) -> Iterator[PostgresSaver]:
    with PostgresSaver.from_conn_string(db_url) as saver:
        saver.serde = JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)
        saver.setup()  # Idempotente: crea las tablas solo si faltan
        yield saver
