"""Lectura de CVs y ofertas desde archivos: se extrae solo el texto (sin imágenes ni diseño)."""

import re
from pathlib import Path

SUPPORTED_EXTENSIONS = (".txt", ".md", ".pdf", ".docx")


class DocumentLoadError(ValueError):
    """El archivo no existe, no es de un formato soportado o no contiene texto extraíble"""


def normalize_whitespace(text: str) -> str:
    """Quita espacios sobrantes por línea y deja como máximo una línea en blanco seguida"""
    lines = [re.sub(r"[ \t ]+", " ", line).strip() for line in text.splitlines()]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _read_docx(path: Path) -> str:
    from docx import Document

    document = Document(path)
    parts = [p.text for p in document.paragraphs]
    # Muchos CVs usan tablas para maquetar columnas: se lee cada celda una sola vez
    for table in document.tables:
        for row in table.rows:
            seen = []
            for cell in row.cells:
                # Las celdas combinadas se repiten en python-docx
                if cell.text not in seen:
                    seen.append(cell.text)
            parts.append(" | ".join(t for t in seen if t.strip()))
    return "\n".join(parts)


def load_text(path: str | Path) -> str:
    """Devuelve el texto plano de un .txt, .md, .pdf o .docx"""
    path = Path(path)
    if not path.is_file():
        raise DocumentLoadError(f"No existe el archivo: {path}")

    extension = path.suffix.lower()
    if extension in (".txt", ".md"):
        text = path.read_text(encoding="utf-8-sig")
    elif extension == ".pdf":
        text = _read_pdf(path)
    elif extension == ".docx":
        text = _read_docx(path)
    else:
        raise DocumentLoadError(
            f"Formato no soportado: '{extension}'. Use uno de: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    text = normalize_whitespace(text)
    if not text:
        hint = " Si es un PDF escaneado (solo imágenes), no tiene texto extraíble; no se aplica OCR." if extension == ".pdf" else ""
        raise DocumentLoadError(f"El archivo {path.name} no contiene texto extraíble.{hint}")
    return text


def read_input(value: str | Path) -> str:
    """Acepta una ruta a un archivo soportado o el texto directamente"""
    if isinstance(value, Path):
        return load_text(value)
    candidate = value.strip()
    looks_like_path = "\n" not in candidate and candidate.lower().endswith(SUPPORTED_EXTENSIONS)
    if looks_like_path:
        return load_text(candidate)
    return normalize_whitespace(value)
