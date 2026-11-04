"""Utilidades de texto sin dependencias del resto del paquete."""

import re
import unicodedata

# Palabras funcionales frecuentes: bastan para distinguir español de inglés en una oferta o un CV
STOPWORDS = {
    "es": {"de", "la", "el", "en", "y", "los", "las", "del", "con", "para", "por", "una", "un", "que", "se",
           "experiencia", "conocimientos", "requisitos", "años", "trabajo"},
    "en": {"the", "and", "of", "to", "in", "with", "for", "a", "an", "is", "are", "on", "you", "we", "our",
           "experience", "requirements", "years", "skills", "work"},
}

def normalize(text: str) -> str:
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
