"""
Cliente del evaluador gratuito de OpenRouter.

Usa ``meta-llama/llama-3-8b-instruct:free`` a través de la API REST nativa
(vía ``requests``), sin dependencias de LangChain. Los modelos gratuitos
OpenRouter no soportan ``response_schema`` ni structured output robusto, por
lo que el prompt se limita a un formato estricto de DOS líneas que se parsea
de forma tolerant.
"""

from __future__ import annotations

import os
import re

import requests

from cvertidor.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_EVAL_MODEL,
)
from cvertidor.models import Evaluation
from cvertidor.prompts import build_evaluator_prompts


def _parse_verdict(raw: str) -> tuple[bool, str]:
    """
    Parsea robustamente la salida del modelo free.

    Espera un bloque tipo::

        VEREDICTO: SÍ
        MOTIVO: <una frase breve>

    Es tolerante a variaciones de mayúsculas/acentos y a salidas parciales.
    """
    upper = raw.upper().replace("´", "´").replace("í", "Í")  # normaliza acentos

    # 1) Intento por línea "VEREDICTO: SÍ/NO"
    m = re.search(r"VEREDICTO\s*[:\-]?\s*([A-ZÓÍÑ ]+)", upper)
    if m:
        token = m.group(1).strip()
        is_faithful = token.startswith(("SÍ", "SI")) and "NO" not in token

        # Extraer motivo si está disponible
        motive = ""
        m2 = re.search(r"MOTIVO\s*[:\-]?\s*(.+)", raw, re.IGNORECASE | re.DOTALL)
        if m2:
            motive = m2.group(1).strip().splitlines()[0][:200]
        return is_faithful, motive or "sin especificar en la salida"

    # 2) Fallback: buscar la primera aparición de un veredicto claro.
    # "SÍ" o "SI" (sin acento) = fiel; "NO" = no fiel.
    if "SÍ" in raw or "SI" in raw:
        return True, "veredicto ambiguo, asumido por presencia de SÍ/SI"
    if "NO" in raw:
        return False, "veredicto ambiguo, asumido por presencia de NO"

    return False, "no se pudo parsear el veredicto del modelo"


def evaluate_cv_faithfulness(
    context: str,
    job_description: str,
    cv_text: str,
    temperature: float = 0.0,
    max_tokens: int = 160,
    timeout: int = 60,
    model: str = OPENROUTER_EVAL_MODEL,
) -> Evaluation:
    """
    Verifica si un CV adaptado es fiel al contexto del candidato original.

    Llama a OpenRouter (modelo free) y devuelve un ``Evaluation`` con el veredicto
    y la justificación. En caso de error en la API, devuelva ``is_faithful=False``
    con una razón explícita para forzar una corrección.
    """
    api_key = OPENROUTER_API_KEY
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY no está configurada en el entorno.")

    system, user = build_evaluator_prompts(context, job_description, cv_text)

    url = f"{OPENROUTER_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # OpenRouter permite rastrear apps vía referer/x-title (opcional).
        "HTTP-Referer": os.getenv("REF_SITE", "https://github.com/cvertidor/cvertidor"),
        "X-Title": os.getenv("APP_TITLE", "CVertidor"),
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        raw = data["choices"][0]["message"]["content"]
    except requests.RequestException as exc:
        return Evaluation(
            is_faithful=False,
            reason=f"Error de red/API: {exc}. Se fuerza corrección por defecto seguro.",
        )
    except (KeyError, IndexError) as exc:
        return Evaluation(
            is_faithful=False,
            reason=f"Respuesta inesperada del modelo: {exc}. Se força corrección.",
        )

    is_faithful, reason = _parse_verdict(raw)
    return Evaluation(is_faithful=is_faithful, reason=reason)
