"""Ejecuta la CLI real, pero agrega una habilidad inventada al primer borrador para forzar un
rechazo de los chequeos deterministas. Los modelos y la validación son los reales."""
import sys
from pathlib import Path

from cvertidor import cli
from cvertidor.graph import nodes

SKILL = "Kubernetes"
original = nodes.generate_adapted_cv
attempt = 0


def generate_with_invention(*args, **kwargs):
    global attempt
    attempt += 1
    draft, raw = original(*args, **kwargs)
    if attempt == 1:
        draft.technical_skills = [*(draft.technical_skills or []), SKILL]
    out = Path("output")
    out.mkdir(exist_ok=True)
    (out / f"intento_{attempt}.json").write_text(draft.model_dump_json(indent=2), encoding="utf-8")
    return draft, raw


nodes.generate_adapted_cv = generate_with_invention
raise SystemExit(cli.main(sys.argv[1:]))
