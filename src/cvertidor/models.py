"""
Esquemas de dominio (Pydantic) de CVertidor.

Se exportan sin modificaciones de estructura respecto al diseño original, para
mantener compatibilidad con el pipeline de RAG y el parser de Gemini.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Certification(BaseModel):
    """Certificación, licencia o acreditación del candidato."""

    name: str = Field(description="Nombre de la certificación, licencia o acreditación.")
    issuing_organization: Optional[str] = Field(
        None, description="Organización que emitió la certificación."
    )
    issue_date: Optional[str] = Field(None, description="Fecha de emisión u vigencia.")


class WorkExperience(BaseModel):
    """Bloque de experiencia laboral con sus responsabilidades y logros."""

    company: str = Field(description="Empresa u organización donde trabajó.")
    role: str = Field(description="Cargo o puesto desempeñado.")
    duration: Optional[str] = Field(
        None, description="Período de trabajo, inicio y término."
    )
    responsibilities: List[str] = Field(
        default_factory=list,
        description="Responsabilidades y logros del cargo.",
    )


class Education(BaseModel):
    """Formación académica del candidato."""

    institution: str = Field(description="Universidad u otra institución educativa.")
    degree: str = Field(description="Título, grado, carrera o programa académico.")
    year: Optional[str] = Field(None, description="Año o período de graduación.")


class StandardCV(BaseModel):
    """Representación canónica y estructurada de un currículum vitae."""

    candidate_name: str = Field(description="Nombre completo del candidato")
    email: Optional[str] = Field(default=None, description="Email")
    phone: Optional[str] = Field(default=None, description="Teléfono de contacto")
    location: Optional[str] = Field(
        default=None, description="Ciudad, región o país de ubicación."
    )
    linkedin_url: Optional[str] = Field(default=None, description="URL de perfil de LinkedIn")
    portfolio_url: Optional[str] = Field(
        default=None, description="URL de GitHub o portafolio"
    )

    education: List[Education] = Field(
        default_factory=list, description="Formación académica."
    )
    work_history: List[WorkExperience] = Field(
        default_factory=list, description="Experiencia laboral."
    )

    technical_skills: Optional[List[str]] = Field(
        default=None, description="Habilidades técnicas."
    )
    certifications: Optional[List[Certification]] = Field(
        default=None, description="Certificaciones o licencias"
    )
    languages: Optional[List[str]] = Field(
        default=None, description="Idiomas y nivel de dominio."
    )
    interests: Optional[List[str]] = Field(
        default=None, description="Intereses y actividades extracurriculares."
    )


# Esquema auxiliar para el evaluador libre de OpenRouter.
# No se usa structured output (el modelo free lo soporta mal), por eso es
# interpretativo y robusto ante salidas poco disciplinadas.
class Evaluation(BaseModel):
    """Resultado de la verificación de fidelidad del CV adaptado."""

    is_faithful: bool = Field(description="True si el CV adaptado es fiel al contexto.")
    reason: str = Field(
        default="",
        description="Justificación breve del veredicto (para retroalimentación).",
    )
