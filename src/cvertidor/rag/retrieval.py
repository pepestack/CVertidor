"""Recuperación de los fragmentos del CV relevantes para la oferta y armado del contexto."""

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import EmbeddingsFilter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

# Secciones que compiten por relevancia con la oferta
RETRIEVED_SECTIONS = ["work_history", "technical_skills", "certifications", "interests"]
# Secciones que siempre van al contexto (no dependen del filtro de similitud)
ALWAYS_INCLUDED_SECTIONS = ["education", "languages"]

# Umbral de similitud del filtro por modelo de embeddings: cada modelo tiene su propia
# escala de similitud coseno, por eso se calibra por separado
SIMILARITY_THRESHOLDS = {
    "google": 0.65,
    "minilm": 0.3,
    # Calibrado con el CV y la oferta de samples/: lo relevante queda en 0.42-0.50 y lo
    # ajeno (BPMN, PowerPoint, intereses, idiomas) bajo 0.30
    "nemotron_3_5_embed_1b": 0.35,
}
DEFAULT_SIMILARITY_THRESHOLD = 0.3


def similarity_threshold(embedding_slug: str) -> float:
    return SIMILARITY_THRESHOLDS.get(embedding_slug, DEFAULT_SIMILARITY_THRESHOLD)


def build_retriever(
    vector_store: VectorStore,
    embeddings: Embeddings,
    candidate_id: str,
    threshold: float,
) -> ContextualCompressionRetriever:
    # Recuperador con MMR para evitar redundancias, filtrado por candidato
    base_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 15,             # Número de chunks finales a recuperar (cada responsabilidad es un chunk)
            "fetch_k": 40,       # Candidatos iniciales a traer de pgvector para evaluar diversidad
            "lambda_mult": 0.7,  # Balance: 1.0 = similitud pura, 0.0 = máxima diversidad (0.5 - 0.7 es ideal)
            "filter": {
                "candidate_id": candidate_id,
                "section": {"$in": RETRIEVED_SECTIONS},
            },
        },
    )

    # Filtrado con EmbeddingsFilter (menor costo que comprimir con un LLM)
    return ContextualCompressionRetriever(
        base_compressor=EmbeddingsFilter(embeddings=embeddings, similarity_threshold=threshold),
        base_retriever=base_retriever,
    )


def format_docs(docs: list[Document]) -> str:
    """Une los chunks en el contexto: descarta repetidos y agrupa las responsabilidades por cargo"""
    seen = set()
    roles = {}   # (empresa, cargo) -> (duración, [responsabilidades])
    others = []

    for doc in docs:
        if doc.page_content in seen:
            continue
        seen.add(doc.page_content)

        if doc.metadata.get("section") == "work_history":
            key = (doc.metadata.get("company"), doc.metadata.get("role"))
            _, bullets = roles.setdefault(key, (doc.metadata.get("duration"), []))
            bullet = doc.page_content.partition("Responsabilidades:\n")[2]
            if bullet:
                bullets.append(bullet)
        else:
            others.append(doc)

    parts = []
    for (company, role), (duration, bullets) in roles.items():
        lines = [f"Rol: {role} - {company}", f"Duración: {duration or 'N/A'}"]
        if bullets:
            lines += ["Responsabilidades:", *bullets]
        parts.append("--- Chunk (work_history) ---\n" + "\n".join(lines))

    for doc in others:
        parts.append(f"--- Chunk ({doc.metadata.get('section', 'N/A')}) ---\n{doc.page_content}")

    return "\n\n".join(parts)
