"""Requisitos estructurados de una oferta de trabajo (los extrae el nodo analyze_job)."""

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from cvertidor.schemas.cv import _empty_to_none, _lenient_list

# Formas en que los modelos escriben el idioma además del código ISO
LANGUAGE_ALIASES = {
    "es": "es", "español": "es", "espanol": "es", "spanish": "es", "castellano": "es",
    "en": "en", "inglés": "en", "ingles": "en", "english": "en",
}


class JobRequirements(BaseModel):
    title: Optional[str] = Field(None, description="Cargo ofrecido, tal como aparece en la oferta.")
    must_have: List[str] = Field(
        default_factory=list,
        description="Requisitos obligatorios: habilidades, herramientas o conocimientos, cada uno en 1 a 4 palabras.",
    )
    nice_to_have: List[str] = Field(
        default_factory=list,
        description="Requisitos deseables u opcionales, cada uno en 1 a 4 palabras.",
    )
    seniority: Optional[str] = Field(None, description="Nivel de experiencia pedido (ej. junior, semi senior, senior).")
    language: Literal["es", "en"] = Field("es", description="Idioma en que está redactada la oferta: 'es' o 'en'.")

    @field_validator("title", "seniority", mode="before")
    @classmethod
    def _optional_text(cls, value: Any) -> Any:
        return _empty_to_none(value)

    @field_validator("must_have", "nice_to_have", mode="before")
    @classmethod
    def _lists(cls, value: Any) -> Any:
        return _lenient_list(value) or []

    @field_validator("language", mode="before")
    @classmethod
    def _language(cls, value: Any) -> Any:
        if isinstance(value, str):
            return LANGUAGE_ALIASES.get(value.strip().lower(), value)
        return value
