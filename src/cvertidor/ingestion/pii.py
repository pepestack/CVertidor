"""Enmascarado de datos personales antes de enviar texto a un modelo externo.

Los proveedores gratuitos pueden registrar los prompts. Correos, teléfonos, RUT y URLs
se reemplazan por marcadores ([EMAIL_1], [TELEFONO_1], ...) y los valores reales se
restauran localmente en el CV resultante.
"""

import re
from dataclasses import dataclass, field

# El orden importa: los correos van primero para que su dominio no se tome como URL
PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")),
    ("URL", re.compile(
        r"(?:https?://|www\.)\S+"
        r"|\b(?:[a-z0-9-]+\.)*(?:linkedin\.com|github\.com|gitlab\.com|behance\.net)/\S*",
        re.IGNORECASE,
    )),
    # RUT chileno: 12.345.678-9, 12345678-K
    ("RUT", re.compile(r"\b\d{1,2}\.?\d{3}\.?\d{3}-[\dkK]\b")),
    # Teléfonos con al menos 8 dígitos: +56 9 6123 4567, (2) 2345 6789, 961234567
    ("TELEFONO", re.compile(r"(?<![\w-])\+?\(?\d[\d\s().-]{7,}\d(?![\w-])")),
]

PLACEHOLDER_RE = re.compile(r"\[(EMAIL|URL|RUT|TELEFONO)_\d+\]")
YEAR_RANGE_RE = re.compile(r"\b(?:19|20)\d{2}\s*[-–]\s*(?:19|20)\d{2}\b")
# Puntuación que suele quedar pegada al final de una URL o correo dentro de una frase
TRAILING_PUNCTUATION = ".,;:)]}"


@dataclass
class MaskResult:
    text: str
    # marcador -> valor original
    mapping: dict[str, str] = field(default_factory=dict)

    def unmask(self, value: str | None) -> str | None:
        return unmask(value, self.mapping)


def _digits(value: str) -> int:
    return sum(c.isdigit() for c in value)


def mask_pii(text: str) -> MaskResult:
    """Reemplaza datos personales por marcadores; el mismo valor recibe siempre el mismo marcador"""
    mapping: dict[str, str] = {}
    reverse: dict[str, str] = {}
    counters: dict[str, int] = {}

    def placeholder_for(kind: str, value: str) -> str:
        if value not in reverse:
            counters[kind] = counters.get(kind, 0) + 1
            token = f"[{kind}_{counters[kind]}]"
            reverse[value] = token
            mapping[token] = value
        return reverse[value]

    for kind, pattern in PATTERNS:
        def replace(match: re.Match) -> str:
            value = match.group(0)
            trailing = ""
            if kind in ("URL", "EMAIL"):
                stripped = value.rstrip(TRAILING_PUNCTUATION)
                trailing = value[len(stripped):]
                value = stripped
            # Evita confundir fechas o rangos de años ("(2015 - 2020)") con teléfonos
            if kind == "TELEFONO" and (_digits(value) < 8 or YEAR_RANGE_RE.search(value)):
                return match.group(0)
            if PLACEHOLDER_RE.fullmatch(value):
                return match.group(0)
            return placeholder_for(kind, value) + trailing

        text = pattern.sub(replace, text)

    return MaskResult(text, mapping)


def unmask(value: str | None, mapping: dict[str, str]) -> str | None:
    """Restaura los valores originales de los marcadores presentes en `value`"""
    if not value:
        return value
    return PLACEHOLDER_RE.sub(lambda m: mapping.get(m.group(0), m.group(0)), value)


def contains_placeholder(value: str | None) -> bool:
    return bool(value) and bool(PLACEHOLDER_RE.search(value))
