"""Resultado de la validación de un CV generado."""

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from cvertidor.schemas.cv import _lenient_list


class Verdict(BaseModel):
    status: Literal["approved", "rejected", "uncertain"]
    # Quién decidió: chequeos deterministas, modelo de decisión o juez
    source: Literal["checks", "decision_model", "judge", "none"] = "none"
    # Probabilidad de cada conducta indebida según el modelo de decisión
    scores: dict[str, float] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)
    # Instrucciones para el siguiente intento de generación
    feedback: str | None = None


class JudgeReview(BaseModel):
    """Salida estructurada del modelo juez"""
    approved: bool = Field(description="true si el CV es fiel al contexto del candidato y cubre lo que el contexto respalda.")
    issues: List[str] = Field(default_factory=list, description="Problemas concretos encontrados, uno por elemento.")
    feedback: Optional[str] = Field(None, description="Si no se aprueba, qué debe corregir el siguiente intento.")

    @field_validator("issues", mode="before")
    @classmethod
    def _issues(cls, value: Any) -> Any:
        return _lenient_list(value) or []
