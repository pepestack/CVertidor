"""Etapas del flujo como funciones: extraer el CV, generar el CV adaptado y guardarlo.

Los nodos del grafo de LangGraph (cvertidor.graph) las orquestan y deciden el orden,
los reintentos y los caminos alternativos.
"""

import hashlib
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from cvertidor.config import Settings
from cvertidor.ingestion.pii import MaskResult, mask_pii, scrub_pii, unmask
from cvertidor.llm.providers import StructuredOutputError, get_structured_model
from cvertidor.prompts.generation import GENERATION_PROMPT, REVISION_BLOCK, VALIDATION_REVISION_BLOCK
from cvertidor.prompts.parsing import PARSING_PROMPT
from cvertidor.schemas.cv import StandardCV

MIN_CV_LENGTH = 50              # Largo minimo de CV
MIN_JOB_DESC_LENGTH = 30        # Largo minimo de oferta de trabajo

# Datos de contacto: se restauran desde los marcadores y deben existir en el CV original
CONTACT_FIELDS = ("email", "phone", "linkedin_url", "portfolio_url")
CANDIDATE_TOKEN = "[CANDIDATO_1]"


class InputError(ValueError):
    """Texto de entrada vacío o demasiado corto"""


class GenerationError(RuntimeError):
    """El modelo no devolvió una salida válida para el esquema pedido"""


@dataclass
class StepMetrics:
    step: str
    seconds: float
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


def validate_text(text: str | None, min_length: int, label: str) -> str:
    """Texto limpio; a diferencia del notebook no se reemplaza por datos de ejemplo"""
    cleaned = (text or "").strip()
    if len(cleaned) < min_length:
        raise InputError(f"{label} está vacío o tiene menos de {min_length} caracteres.")
    return cleaned


def _compact(value: str) -> str:
    """Forma comparable de un dato de contacto: sin esquema, espacios ni separadores"""
    value = re.sub(r"(?i)\b(https?://)?(www\.)?", "", value.strip())
    return re.sub(r"[\s()./:-]", "", value).lower()


def _map_strings(model: BaseModel, fn: Callable[[str], str]) -> BaseModel:
    """Aplica `fn` a todos los campos de texto del modelo"""
    def walk(value: Any) -> Any:
        if isinstance(value, str):
            return fn(value)
        if isinstance(value, list):
            return [walk(v) for v in value]
        if isinstance(value, dict):
            return {k: walk(v) for k, v in value.items()}
        return value

    return type(model).model_validate(walk(model.model_dump()))


def unmask_model(model: BaseModel, mapping: dict[str, str]) -> BaseModel:
    """Restaura los marcadores de PII en todos los campos de texto del modelo"""
    return _map_strings(model, lambda value: unmask(value, mapping))


def mask_cv(cv: StandardCV, mapping: dict[str, str] | None = None) -> tuple[StandardCV, dict[str, str]]:
    """Copia del CV sin datos personales, con una sola numeración de marcadores. Es lo único
    que se indexa o se envía a los modelos después de la extracción. Es determinista: el
    mismo CV produce los mismos marcadores, así que no hay que guardar el mapeo.
    Con `mapping` (el del CV original) un borrador conserva los marcadores de ese CV"""
    mapping = dict(mapping or {})
    masked =_map_strings(cv, lambda value: mask_pii(value, mapping).text.replace(cv.candidate_name, CANDIDATE_TOKEN))
    mapping[CANDIDATE_TOKEN] = cv.candidate_name
    return masked, mapping


def restore_contact(parsed: StandardCV, masked: MaskResult, original_text: str) -> tuple[StandardCV, list[str]]:
    """Restaura los datos de contacto y descarta los que no aparecen en el CV original
    (los modelos gratuitos tienden a inventar correos, teléfonos o perfiles de LinkedIn)"""
    restored = unmask_model(parsed, masked.mapping)
    source = _compact(original_text)
    removed = []
    for name in CONTACT_FIELDS:
        value = getattr(restored, name)
        if value and _compact(value) not in source:
            removed.append(f"{name}={value!r}")
            setattr(restored, name, None)
    return restored, removed


def step_metrics(step: str, start: float, raw: Any = None) -> StepMetrics:
    usage = getattr(raw, "usage_metadata", None) or {}
    model = (getattr(raw, "response_metadata", None) or {}).get("model_name")
    return StepMetrics(step, time.perf_counter() - start, model, usage.get("input_tokens"), usage.get("output_tokens"))


def invoke_structured(chain, inputs: dict, what: str):
    try:
        result = chain.invoke(inputs)
    except StructuredOutputError as error:
        # Ningún modelo de la cadena (principal ni fallbacks) devolvió una salida válida
        raise GenerationError(f"El modelo no devolvió un {what} válido: {error}") from error
    if result["parsed"] is None:
        raw = result["raw"]
        finish_reason = (raw.response_metadata or {}).get("finish_reason")
        raise GenerationError(
            f"El modelo no devolvió un {what} válido (finish_reason={finish_reason}): {result.get('parsing_error')}"
        )
    return result["parsed"], result["raw"]


def parse_cv(cv_text: str, settings: Settings) -> tuple[StandardCV, list[str], Any]:
    """Estructura el CV original. El modelo solo ve el texto con la PII enmascarada"""
    masked = mask_pii(cv_text)
    chain = PARSING_PROMPT | get_structured_model("parser", StandardCV, include_raw=True, settings=settings)
    parsed, raw = invoke_structured(chain, {"cv_text": masked.text}, "StandardCV del CV original")
    restored, removed = restore_contact(parsed, masked, cv_text)
    return restored, removed, raw


def generate_adapted_cv(
    context: str,
    job_text: str,
    parsed_cv: StandardCV,
    settings: Settings,
    plan: str = "",
    current_draft: str | None = None,
    human_edit: bool = False,
) -> tuple[StandardCV, Any]:
    """Con `current_draft` edita ese borrador en vez de reescribir: según las correcciones de una
    persona (`human_edit`) o, si no, según lo que rechazó la validación"""
    chain = GENERATION_PROMPT | get_structured_model("generator", StandardCV, include_raw=True, settings=settings)
    block = REVISION_BLOCK if human_edit else VALIDATION_REVISION_BLOCK
    adapted, raw = invoke_structured(
        chain,
        {"context": context, "job_description": scrub_pii(job_text), "plan": plan or "- Sin instrucciones adicionales.",
         "revision": block.format(draft=current_draft) if current_draft else ""},
        "StandardCV adaptado",
    )
    # El contexto va enmascarado: se restauran los marcadores que el modelo haya copiado
    adapted = unmask_model(adapted, mask_cv(parsed_cv)[1])
    # Nombre y datos de contacto se copian del CV original, no del modelo
    adapted.candidate_name = parsed_cv.candidate_name
    adapted.location = parsed_cv.location
    for name in CONTACT_FIELDS:
        setattr(adapted, name, getattr(parsed_cv, name))
    # Una persona puede haber pedido cambiar un cargo
    if not human_edit:
        restore_roles(adapted, parsed_cv)
    return adapted, raw


def restore_missing_work_history(adapted: StandardCV, original: StandardCV) -> bool:
    """Ante una oferta poco afín los modelos eliminan toda la experiencia, y una instrucción no
    lo corrige. Empresa, cargo y fechas son hechos conocidos: se agregan sin responsabilidades"""
    if adapted.work_history or not original.work_history:
        return False
    adapted.work_history = [w.model_copy(update={"responsibilities": []}) for w in original.work_history]
    return True


def restore_roles(adapted: StandardCV, original: StandardCV) -> None:
    """Los cargos se conservan textualmente aunque el CV se traduzca (los modelos los
    traducen pese a la instrucción); se emparejan por empresa"""
    roles = {_compact(w.company): w.role for w in original.work_history}
    for work in adapted.work_history:
        role = roles.get(_compact(work.company))
        if role:
            work.role = role


def ensure_db_available(db_url: str) -> None:
    """Falla rápido si la base no responde, antes de gastar llamadas a modelos (cupo diario)"""
    import psycopg

    with psycopg.connect(db_url, connect_timeout=5) as conn:
        conn.execute("SELECT 1")


def save_outputs(candidate_id: str, markdown: str, html: str, output_dir: Path, job_text: str = "") -> list[Path]:
    """Los nombres dependen del contenido: guardar lo mismo dos veces no duplica archivos, y un
    CV para otra oferta no sobrescribe el anterior. candidate_id solo contiene [a-z0-9_]"""
    output_dir.mkdir(parents=True, exist_ok=True)
    version = hashlib.sha256(f"{job_text}\n{markdown}".encode("utf-8")).hexdigest()[:8]
    paths = [output_dir / f"cv_{candidate_id}_{version}.md", output_dir / f"cv_{candidate_id}_{version}.html"]
    paths[0].write_text(markdown, encoding="utf-8")
    paths[1].write_text(html, encoding="utf-8")
    return paths
