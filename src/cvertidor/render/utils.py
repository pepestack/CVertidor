"""Utilidades de texto compartidas por los generadores de Markdown y HTML."""

import re
from html import escape as html_escape


def _url(url: str) -> str:
    """Agrega https:// si la URL no trae esquema (ej. 'linkedin.com/in/...')"""
    return url if re.match(r"^https?://", url, re.IGNORECASE) else f"https://{url}"


def _join(*parts, sep: str = " | ") -> str:
    """Une solo las partes con valor (evita imprimir 'None')"""
    return sep.join(p for p in parts if p)


def esc(value) -> str:
    """Escapa texto para insertarlo en HTML"""
    return html_escape(str(value)) if value else ""
