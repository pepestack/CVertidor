"""Primera capa de validación: chequeos deterministas (sin red ni costo) y umbrales.

Lo que se puede calcular se calcula aquí; el modelo de decisión solo responde lo que
requiere juicio. Una habilidad agregada, por ejemplo, apenas sube su probabilidad (medido:
0.14 con 'Kubernetes' agregado), pero este chequeo la detecta siempre.
"""

import json
import re

from pydantic import BaseModel

from cvertidor.ingestion.pii import HIDDEN_DATA, contains_placeholder
from cvertidor.schemas.cv import StandardCV
from cvertidor.text import detect_language, normalize

MAX_RESPONSIBILITIES = 4

# Umbrales del modelo de decisión (probabilidad de la conducta indebida), medidos con Span-01
# Lite sobre el CV de samples/: CV fiel 0.025, datos inventados 0.96, inyección de
# instrucciones 0.965, requisitos omitidos 0.99. Entre ambos valores se consulta al juez.
# Aprobar algo inventado cuesta más que una llamada extra: por eso el corte bajo es estricto
APPROVE_BELOW = {"inventa_datos": 0.15, "omite_requisitos": 0.4}
REJECT_ABOVE = {"inventa_datos": 0.65, "omite_requisitos": 0.65}

_NUMBER_RE = re.compile(r"\d[\d.,]*\d%?|\d%")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", normalize(text))


def _matches(value: str, known: set[str]) -> bool:
    """El valor coincide con uno conocido, o uno contiene al otro (nombres abreviados)"""
    key = _key(value)
    return bool(key) and any(key == k or key in k or k in key for k in known if k)


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9+#]+", normalize(text)) if len(w) > 2]


def _numbers(text: str) -> set[str]:
    """Cifras significativas sin separadores (99,9% y 99.9% son la misma); ignora dígitos sueltos"""
    return {re.sub(r"[.,%]", "", n) for n in _NUMBER_RE.findall(text)}


def _strings(model: BaseModel) -> list[str]:
    found = []

    def walk(value):
        if isinstance(value, str):
            found.append(value)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)

    walk(model.model_dump())
    return found


def run_checks(
    draft: StandardCV,
    source: StandardCV,
    output_language: str,
    translate: bool = False,
    confirmed: list[str] | None = None,
) -> list[str]:
    """Problemas objetivos del CV generado respecto del CV original; vacío si no hay.
    `confirmed`: correcciones de una persona. Su texto cuenta como fuente, y también manda
    sobre las reglas de forma (extensión e idioma), porque pudo pedirlas expresamente"""
    issues = []
    confirmed = confirmed or []
    confirmed_text = " ".join(confirmed)
    source_json = json.dumps(source.model_dump(), ensure_ascii=False)
    source_text = normalize(f"{source_json} {confirmed_text}")
    source_words = set(re.findall(r"[a-z0-9+#]+", source_text))
    source_numbers = _numbers(f"{source_json} {confirmed_text}")
    confirmed_years = set(_YEAR_RE.findall(confirmed_text))
    confirmed_key = _key(confirmed_text)

    companies = {_key(w.company) for w in source.work_history} | {confirmed_key}
    durations = {_key(w.company): w.duration or "" for w in source.work_history}
    for work in draft.work_history:
        if not _matches(work.company, companies):
            issues.append(f"La empresa '{work.company}' no está en el CV original.")
            continue
        original = next((d for c, d in durations.items() if c == _key(work.company)), None)
        if original is not None and not set(_YEAR_RE.findall(work.duration or "")) <= set(_YEAR_RE.findall(original)) | confirmed_years:
            issues.append(f"Las fechas de '{work.company}' ({work.duration}) no coinciden con el CV original ({original}).")
        if not confirmed and len(work.responsibilities) > MAX_RESPONSIBILITIES:
            issues.append(f"'{work.company}' tiene {len(work.responsibilities)} responsabilidades; el máximo es {MAX_RESPONSIBILITIES}.")

    institutions = {_key(e.institution) for e in source.education} | {confirmed_key}
    for edu in draft.education:
        if not _matches(edu.institution, institutions):
            issues.append(f"La institución '{edu.institution}' no está en el CV original.")

    certifications = {_key(c.name) for c in source.certifications or []} | {confirmed_key}
    for cert in draft.certifications or []:
        if not _matches(cert.name, certifications):
            issues.append(f"La certificación '{cert.name}' no está en el CV original.")

    for skill in draft.technical_skills or []:
        words = _words(skill)
        # Traducida, solo parte de la habilidad coincide ("Excel Avanzado" -> "Advanced Excel")
        found = any if translate else all
        if words and not found(w in source_words for w in words):
            issues.append(f"La habilidad '{skill}' no está en el CV original.")

    prose = [draft.professional_summary or ""] + [r for w in draft.work_history for r in w.responsibilities]
    for number in sorted(_numbers(" ".join(prose)) - source_numbers):
        issues.append(f"La cifra '{number}' no está en el CV original.")

    if any(contains_placeholder(s) or HIDDEN_DATA in s for s in _strings(draft)):
        issues.append("El CV generado contiene marcadores de datos personales sin resolver.")

    language = detect_language(" ".join(prose))
    if not confirmed and language and language != output_language:
        issues.append(f"El CV está escrito en '{language}', pero el idioma de salida es '{output_language}'.")
    return issues


def status_from_scores(scores: dict[str, float], human_corrected: bool = False) -> str:
    """Si una persona corrigió el CV, una omisión no se rechaza: pudo quitar el dato a propósito"""
    ignored = {"omite_requisitos"} if human_corrected else set()
    if any(scores.get(name, 0.0) >= limit for name, limit in REJECT_ABOVE.items() if name not in ignored):
        return "rejected"
    if all(scores.get(name, 1.0) <= limit for name, limit in APPROVE_BELOW.items() if name not in ignored):
        return "approved"
    return "uncertain"
