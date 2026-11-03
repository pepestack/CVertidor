"""Utilidades de texto compartidas por los generadores de Markdown y HTML."""

import re
from html import escape as html_escape


# Títulos de sección por idioma de salida (el plan decide el idioma según la oferta)
LABELS = {
    "es": {
        "phone": "Teléfono", "location": "Ubicación", "summary": "Perfil Profesional",
        "work": "Experiencia Laboral", "education": "Educación", "skills": "Habilidades Técnicas",
        "certifications": "Certificaciones", "languages": "Idiomas", "interests": "Intereses",
    },
    "en": {
        "phone": "Phone", "location": "Location", "summary": "Professional Summary",
        "work": "Work Experience", "education": "Education", "skills": "Technical Skills",
        "certifications": "Certifications", "languages": "Languages", "interests": "Interests",
    },
}


def labels(lang: str) -> dict[str, str]:
    return LABELS.get(lang, LABELS["es"])


def _url(url: str) -> str:
    """Agrega https:// si la URL no trae esquema (ej. 'linkedin.com/in/...')"""
    return url if re.match(r"^https?://", url, re.IGNORECASE) else f"https://{url}"


def _join(*parts, sep: str = " | ") -> str:
    """Une solo las partes con valor (evita imprimir 'None')"""
    return sep.join(p for p in parts if p)


def esc(value) -> str:
    """Escapa texto para insertarlo en HTML"""
    return html_escape(str(value)) if value else ""
