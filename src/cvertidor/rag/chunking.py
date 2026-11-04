"""Conversión de un StandardCV en fragmentos (Documents) para la base vectorial."""

import re
import unicodedata
import uuid

from langchain_core.documents import Document

from cvertidor.schemas.cv import StandardCV


def make_candidate_id(name: str) -> str:
    """Identificador estable del candidato: sin tildes, en minúsculas y solo [a-z0-9_]"""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_") or "candidato"


def make_doc_id(collection_name: str, doc: Document) -> str:
    """ID determinista: el mismo chunk en la misma colección siempre genera el mismo ID.
    Incluye la colección porque el id de langchain_pg_embedding es único entre todas las colecciones."""
    key = f"{collection_name}|{doc.metadata['candidate_id']}|{doc.metadata['section']}|{doc.page_content}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def create_cv_documents(parsed_cv: StandardCV, candidate_id: str | None = None) -> list[Document]:
    """`candidate_id` es obligatorio si el CV viene enmascarado (el nombre ya no sirve para derivarlo)"""
    candidate_id = candidate_id or make_candidate_id(parsed_cv.candidate_name)
    candidate_name = parsed_cv.candidate_name
    documents = []

    # 1. Experiencia laboral: un chunk por responsabilidad, cada uno con el encabezado del cargo
    for exp in parsed_cv.work_history:

        role_header = (
            f"Candidato: {candidate_name}\n"
            f"Rol: {exp.role} - {exp.company}\n"
            f"Duración: {exp.duration or 'N/A'}"
        )

        base_metadata = {
            "candidate_id": candidate_id,
            "section": "work_history",
            "company": exp.company,
            "role": exp.role,
            "duration": exp.duration,
        }

        responsibilities = [r.strip() for r in exp.responsibilities if r.strip()]

        # Cargo sin responsabilidades: se guarda solo el encabezado para no perderlo
        if not responsibilities:
            documents.append(
                Document(
                    page_content=role_header,
                    metadata={**base_metadata, "sub_chunk_index": 0},
                )
            )
            continue

        for i, resp in enumerate(responsibilities):
            documents.append(
                Document(
                    page_content=f"{role_header}\nResponsabilidades:\n- {resp}",
                    metadata={**base_metadata, "sub_chunk_index": i},
                )
            )

    # 2. Chunk de Educacion
    for edu in parsed_cv.education:
        content = (
            f"Candidato: {candidate_name}\n"
            f"Educacion: {edu.degree} - {edu.institution}\n"
            f"Año: {edu.year or 'N/A'}"
        )

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "candidate_id": candidate_id,
                    "section": "education",
                    "institution": edu.institution,
                },
            )
        )

    # 3. Chunk de Habilidades
    if parsed_cv.technical_skills:
        documents.append(
            Document(
                page_content=(
                    f"Candidato: {candidate_name}\n"
                    f"Habilidades técnicas: {', '.join(parsed_cv.technical_skills)}"
                ),
                metadata={
                    "candidate_id": candidate_id,
                    "section": "technical_skills",
                },
            )
        )

    # 4. Certificaciones
    if parsed_cv.certifications:
        certs = "; ".join(
            f"{c.name} ({c.issuing_organization or 'N/A'}) {c.issue_date or ''}".strip()
            for c in parsed_cv.certifications
        )

        documents.append(
            Document(
                page_content=(
                    f"Candidato: {candidate_name}\n"
                    f"Certificaciones: {certs}"
                ),
                metadata={
                    "candidate_id": candidate_id,
                    "section": "certifications",
                },
            )
        )

    # 5. Idiomas
    if parsed_cv.languages:
        langs = ", ".join(parsed_cv.languages)
        documents.append(
            Document(
                page_content=(
                    f"Candidato: {candidate_name}\n"
                    f"Idiomas: {langs}"
                ),
                metadata={
                    "candidate_id": candidate_id,
                    "section": "languages",
                },
            )
        )

    # 6. Intereses
    if parsed_cv.interests:
        interests_list = "\n- ".join(parsed_cv.interests)
        documents.append(
            Document(
                page_content=(
                    f"Candidato: {candidate_name}\n"
                    f"Intereses:\n- {interests_list}"
                ),
                metadata={
                    "candidate_id": candidate_id,
                    "section": "interests",
                },
            )
        )

    return documents
