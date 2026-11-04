"""Planificador determinista del nodo `plan`: decide el idioma de salida, el orden de las
secciones y las consultas de recuperación a partir de la oferta y del CV estructurado.

No usa un LLM: las decisiones son reproducibles y se pueden probar sin red.
"""

import re
import unicodedata

from pydantic import BaseModel, Field

from cvertidor.schemas.cv import StandardCV
from cvertidor.schemas.job import JobRequirements

LANGUAGE_NAMES = {"es": "español", "en": "inglés"}

# Palabras funcionales frecuentes: bastan para distinguir español de inglés en una oferta o un CV
STOPWORDS = {
    "es": {"de", "la", "el", "en", "y", "los", "las", "del", "con", "para", "por", "una", "un", "que", "se",
           "experiencia", "conocimientos", "requisitos", "años", "trabajo"},
    "en": {"the", "and", "of", "to", "in", "with", "for", "a", "an", "is", "are", "on", "you", "we", "our",
           "experience", "requirements", "years", "skills", "work"},
}

# Con menos de esta fracción de requisitos obligatorios presentes en el CV, el plan
# destaca habilidades y formación por sobre la experiencia
LOW_COVERAGE = 0.5

DEFAULT_PRIORITY = ["work_history", "technical_skills", "certifications", "education"]
LOW_COVERAGE_PRIORITY = ["technical_skills", "certifications", "education", "work_history"]

SECTION_NAMES = {
    "work_history": "experiencia laboral",
    "technical_skills": "habilidades técnicas",
    "certifications": "certificaciones",
    "education": "educación",
}


class Plan(BaseModel):
    output_language: str = Field(description="Idioma del CV adaptado: 'es' o 'en'.")
    cv_language: str
    translate: bool
    skill_coverage: float | None = Field(None, description="Fracción de requisitos obligatorios presentes en el CV.")
    matched_requirements: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    section_priority: list[str] = Field(default_factory=lambda: list(DEFAULT_PRIORITY))
    # La primera consulta es la oferta completa; las siguientes se usan si la recuperación
    # no encuentra experiencia laboral relevante
    retrieval_queries: list[str] = Field(default_factory=list)
    instructions: str = ""


def _normalize(text: str) -> str:
    """Minúsculas y sin tildes, para comparar requisitos con el texto del CV"""
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def detect_language(text: str) -> str | None:
    """'es' o 'en' según las palabras funcionales; None si no hay suficiente evidencia"""
    words = re.findall(r"[a-záéíóúñ]+", text.lower())
    counts = {lang: sum(w in stop for w in words) for lang, stop in STOPWORDS.items()}
    es, en = counts["es"], counts["en"]
    if max(es, en) < 3 or es == en:
        return None
    return "es" if es > en else "en"


def cv_haystack(cv: StandardCV) -> str:
    """Texto del CV donde se buscan los requisitos de la oferta"""
    parts = [cv.professional_summary or ""]
    parts += cv.technical_skills or []
    parts += [c.name for c in cv.certifications or []]
    for work in cv.work_history:
        parts += [work.role, *work.responsibilities]
    parts += [e.degree for e in cv.education]
    parts += cv.languages or []
    return _normalize(" ".join(parts))


def requirement_in_cv(requirement: str, haystack: str) -> bool:
    """El requisito aparece completo o todas sus palabras significativas aparecen en el CV"""
    req = _normalize(requirement).strip()
    if not req:
        return False
    if re.search(rf"(?<![a-z0-9]){re.escape(req)}(?![a-z0-9])", haystack):
        return True
    words = [w for w in re.findall(r"[a-z0-9+#]+", req) if len(w) > 2]
    return bool(words) and all(re.search(rf"(?<![a-z0-9]){re.escape(w)}(?![a-z0-9])", haystack) for w in words)


def build_plan(
    cv: StandardCV,
    job: JobRequirements,
    job_text: str,
    cv_text: str = "",
    preferences: list[str] | None = None,
) -> Plan:
    # El idioma detectado en el texto manda; el que informó el modelo solo desempata
    output_language = detect_language(job_text) or job.language
    cv_language = detect_language(cv_text or cv_haystack(cv)) or "es"

    haystack = cv_haystack(cv)
    matched = [r for r in job.must_have if requirement_in_cv(r, haystack)]
    missing = [r for r in job.must_have if r not in matched]
    coverage = len(matched) / len(job.must_have) if job.must_have else None
    low_coverage = coverage is not None and coverage < LOW_COVERAGE
    priority = list(LOW_COVERAGE_PRIORITY if low_coverage else DEFAULT_PRIORITY)

    queries = [job_text]
    focus = job.must_have or job.nice_to_have
    if focus:
        queries.append(f"{job.title or 'Cargo'}: experiencia con {', '.join(focus)}")
    if job.title:
        queries.append(job.title)

    lines = [f"Idioma de salida: {LANGUAGE_NAMES.get(output_language, output_language)}."]
    if output_language != cv_language:
        lines.append(f"Traduce el contenido del candidato del {LANGUAGE_NAMES.get(cv_language, cv_language)} "
                     f"al {LANGUAGE_NAMES.get(output_language, output_language)}.")
    lines.append("Orden de prioridad de las secciones: " + ", ".join(SECTION_NAMES[s] for s in priority) + ".")
    if matched:
        lines.append("Requisitos de la oferta que el candidato sí demuestra (destácalos): " + ", ".join(matched) + ".")
    if missing:
        lines.append("Requisitos que no aparecen en el contexto del candidato (no los agregues ni los insinúes): "
                     + ", ".join(missing) + ".")
    if low_coverage:
        lines.append("El candidato cumple pocos requisitos obligatorios: destaca habilidades transferibles "
                     "y formación respaldadas por el contexto.")
    if preferences:
        lines.append("Preferencias recordadas del usuario: " + "; ".join(preferences) + ".")

    return Plan(
        output_language=output_language,
        cv_language=cv_language,
        translate=output_language != cv_language,
        skill_coverage=coverage,
        matched_requirements=matched,
        missing_requirements=missing,
        section_priority=priority,
        retrieval_queries=queries,
        instructions="\n".join(f"- {line}" for line in lines),
    )
