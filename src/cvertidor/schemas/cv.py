"""Modelos de Pydantic para validar la estructura de un currículum."""

from typing import List, Optional

from pydantic import BaseModel, Field


class Certification(BaseModel):
    name: str = Field(description="Nombre de la certificación, licencia o acreditación.")
    issuing_organization: Optional[str] = Field(None, description="Organización que emitió la certificación o licencia.")
    issue_date: Optional[str] = Field(None, description="Fecha de emisión, obtención o vigencia indicada.")

class WorkExperience(BaseModel):
    company: str = Field(description="Empresa u organización donde trabajó.")
    role: str = Field(description="Cargo o puesto desempeñado.")
    duration: Optional[str] = Field(None, description="Período de trabajo, incluyendo inicio y término si aparecen.")
    responsibilities: List[str] = Field(default_factory=list, description="Responsabilidades, tareas, contribuciones y logros del cargo.")

class Education(BaseModel):
    institution: str = Field(description="Universidad u otra Institución educativa.")
    degree: str = Field(description="Título, grado, carrera o programa académico.")
    year: Optional[str] = Field(None, description="Año o período de graduación, egreso o finalización.")


# 2. CV Estandar
class StandardCV(BaseModel):
    # Header o Encabezado
    candidate_name: str = Field(description="Nombre completo de candidato")
    email: Optional[str] = Field(default=None, description="Email")
    phone: Optional[str] = Field(default=None, description="Telefono de contacto")
    location: Optional[str] = Field(default=None, description="Ciudad, región o país indicado como ubicación.")
    linkedin_url: Optional[str] = Field(default=None, description="URL de perfil de LinkedIn")
    portfolio_url: Optional[str] = Field(default=None, description="URL de GitHub o portafolio")

    professional_summary: Optional[str] = Field(default=None, description="Resumen profesional breve y adaptado a la oferta, basado únicamente en información explícita del candidato.")

    # Experiencia y educación
    education: List[Education] = Field(default_factory=list, description="Formación académica del candidato.")
    work_history: List[WorkExperience] = Field(default_factory=list, description="Experiencia laboral y profesional.")

    # Secciones opcionales
    technical_skills: Optional[List[str]] = Field(
        default=None,
        description="Habilidades. ej, lenguajes de programación, frameworks u otras herramientas."
    )
    certifications: Optional[List[Certification]] = Field(
        default=None,
        description="Certificaciones o licencias"
    )
    languages: Optional[List[str]] = Field(
        default=None,
        description="Idiomas y nivel de dominio cuando esté indicado."
    )
    interests: Optional[List[str]] = Field(
        default=None,
        description="Intereses y actividades extracurriculares relevantes."
    )
