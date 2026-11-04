"""
Procesamiento y chunkeo de documentos del currículum.

Convierte un ``StandardCV`` en una lista de ``Document`` de LangChain, aplicando
un chunkeo por secciones que permite una recuperación vectorial (MMR) más precisa.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from cvertidor.models import StandardCV

# Splitter compartido en todo el módulo.
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n- ", "\n* ", "\n", ". ", " "],
)


def create_cv_documents(parsed_cv: StandardCV) -> list[Document]:
    """
    Transforma un ``StandardCV`` en ``Document`` listos para insertar en pgvector.

    - La experiencia laboral con responsabilidades largas se subdivide, conservando
      un encabezado con contexto (candidato, rol, empresa, duración).
    - Cada sección (educación, habilidades, certificaciones, idiomas, intereses)
      se empaqueta en su propio ``Document`` con metadata de trazabilidad.
    """
    candidate_id = parsed_cv.candidate_name.lower().replace(" ", "_")
    candidate_name = parsed_cv.candidate_name
    documents: list[Document] = []

    # --- Experiencia laboral --------------------------------------------- #
    for exp in parsed_cv.work_history:
        header = (
            f"Candidato: {candidate_name}\n"
            f"Rol: {exp.role} - {exp.company}\n"
            f"Duración: {exp.duration or 'N/A'}\n"
            "Responsabilidades:\n"
        )
        resp = "\n- ".join(exp.responsibilities)
        resp_content = f"- {resp}"
        base_meta = {
            "candidate_id": candidate_id,
            "section": "work_history",
            "company": exp.company,
            "role": exp.role,
        }

        if len(resp_content) <= 500:
            documents.append(
                Document(
                    page_content=header + resp_content,
                    metadata={**base_meta, "sub_chunk_index": 0},
                )
            )
        else:
            temp = Document(page_content=resp_content, metadata=base_meta)
            for i, sub in enumerate(TEXT_SPLITTER.split_documents([temp])):
                sub.page_content = header + sub.page_content
                sub.metadata["sub_chunk_index"] = i
                documents.append(sub)

    # --- Educación ------------------------------------------------------- #
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

    # --- Habilidades técnicas ------------------------------------------- #
    if parsed_cv.technical_skills:
        documents.append(
            Document(
                page_content=(
                    f"Candidato: {candidate_name}\n"
                    f"Habilidades técnicas: {', '.join(parsed_cv.technical_skills)}"
                ),
                metadata={"candidate_id": candidate_id, "section": "technical_skills"},
            )
        )

    # --- Certificaciones ----------------------------------------------- #
    if parsed_cv.certifications:
        certs = "; ".join(
            f"{c.name} ({c.issuing_organization or 'N/A'}) {c.issue_date or ''}".strip()
            for c in parsed_cv.certifications
        )
        documents.append(
            Document(
                page_content=f"Candidato: {candidate_name}\nCertificaciones: {certs}",
                metadata={"candidate_id": candidate_id, "section": "certifications"},
            )
        )

    # --- Idiomas ------------------------------------------------------- #
    if parsed_cv.languages:
        documents.append(
            Document(
                page_content=(
                    f"Candidato: {candidate_name}\n"
                    f"Idiomas: {', '.join(parsed_cv.languages)}"
                ),
                metadata={"candidate_id": candidate_id, "section": "languages"},
            )
        )

    # --- Intereses ----------------------------------------------------- #
    if parsed_cv.interests:
        interests_list = "\n- ".join(parsed_cv.interests)
        documents.append(
            Document(
                page_content=f"Candidato: {candidate_name}\nIntereses:\n- {interests_list}",
                metadata={"candidate_id": candidate_id, "section": "interests"},
            )
        )

    return documents


def candidate_id_from_name(name: str) -> str:
    """Deriva un ``candidate_id`` estable a partir del nombre del candidato."""
    return name.lower().replace(" ", "_")
