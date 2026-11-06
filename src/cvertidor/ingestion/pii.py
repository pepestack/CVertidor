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
    # Teléfonos con al menos 8 dígitos: +56 9 6123 4567, (+56) 9 6123 4567, (2) 2345 6789, 961234567.
    # En una sola línea: dos líneas seguidas con números (fechas de un CV) no son un teléfono
    ("TELEFONO", re.compile(r"(?<![\w-])\(?\+?\(?\d(?:[\d().-]|[^\S\r\n]){7,}\d(?![\w-])")),
]

PLACEHOLDER_RE = re.compile(r"\[(EMAIL|URL|RUT|TELEFONO|CANDIDATO)_\d+\]")
# Marcador de los datos de una oferta: no se restauran y no pueden confundirse con los del CV
HIDDEN_DATA = "[DATO_OCULTO]"
# Solo años y separadores: "(2015 - 2020)", "2022 2015". Debe calzar el valor completo, porque un
# teléfono real puede contener grupos que parecen años (+56 2 2012 2013)
YEARS_ONLY_RE = re.compile(r"[\s().-]*(?:(?:19|20)\d{2}[\s().-]*){2,}")
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


def mask_pii(text: str, mapping: dict[str, str] | None = None) -> MaskResult:
    """Reemplaza datos personales por marcadores; el mismo valor recibe siempre el mismo marcador.
    Con `mapping` la numeración se comparte entre varias llamadas (y el diccionario se actualiza)"""
    mapping = {} if mapping is None else mapping
    reverse = {value: token for token, value in mapping.items()}
    counters: dict[str, int] = {}
    for token in mapping:
        kind = token[1:].split("_")[0]
        counters[kind] = counters.get(kind, 0) + 1

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
            leading = trailing = ""
            if kind in ("URL", "EMAIL"):
                stripped = value.rstrip(TRAILING_PUNCTUATION)
                trailing = value[len(stripped):]
                value = stripped
            # Evita confundir fechas o rangos de años ("(2015 - 2020)") con teléfonos
            if kind == "TELEFONO" and (_digits(value) < 8 or YEARS_ONLY_RE.fullmatch(value)):
                return match.group(0)
            # El paréntesis que envuelve al número, como en "(+56 9 1234 5678)", no es parte de él
            if kind == "TELEFONO" and value.startswith("(") and value.count("(") > value.count(")"):
                leading, value = "(", value[1:]
            if PLACEHOLDER_RE.fullmatch(value):
                return match.group(0)
            return leading + placeholder_for(kind, value) + trailing

        text = pattern.sub(replace, text)

    return MaskResult(text, mapping)


def scrub_pii(text: str) -> str:
    """Oculta los datos personales de un texto cuyo original no hay que restaurar (una oferta)"""
    return PLACEHOLDER_RE.sub(HIDDEN_DATA, mask_pii(text).text)


def unmask(value: str | None, mapping: dict[str, str]) -> str | None:
    """Restaura los valores originales de los marcadores presentes en `value`"""
    if not value:
        return value
    return PLACEHOLDER_RE.sub(lambda m: mapping.get(m.group(0), m.group(0)), value)


def contains_placeholder(value: str | None) -> bool:
    return bool(value) and bool(PLACEHOLDER_RE.search(value))
