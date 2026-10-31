import httpx
import openai

from cvertidor import cli
from cvertidor.llm import check
from cvertidor.llm.check import CheckResult, describe_error, roles_without_working_model


def test_describe_error_con_codigos_conocidos():
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    error = openai.NotFoundError("no", response=httpx.Response(404, request=request), body=None)

    assert describe_error(error) == "modelo no encontrado (404)"
    assert describe_error(ValueError("x" * 300)).startswith("ValueError: xxx")
    assert len(describe_error(ValueError("x" * 300))) < 200


def test_prueba_cada_modelo_una_sola_vez(make_settings, monkeypatch):
    calls = []

    def fake_structured(spec, role, method, settings):
        calls.append(str(spec))
        if spec.model == "caido":
            raise ConnectionError("sin respuesta")
        return "salida estructurada"

    monkeypatch.setattr(check, "_check_structured", fake_structured)
    monkeypatch.setattr(check, "_check_tools", lambda spec, role, settings: "llamada a herramientas")
    settings = make_settings(parser_model="caido", parser_model_fallbacks="ok")

    results = check.check_roles(["parser", "job", "refiner"], settings=settings, capabilities={})

    # job comparte modelos con parser: no se repiten las llamadas
    assert calls == ["openrouter:caido", "openrouter:ok"]
    parser = [r for r in results if r.role == "parser"]
    assert [(r.ok, r.is_fallback) for r in parser] == [(False, False), (True, True)]
    assert roles_without_working_model(results) == []


def test_roles_sin_modelo_disponible():
    results = [
        CheckResult("parser", "a", "structured", False, "", 0),
        CheckResult("parser", "b", "structured", True, "", 0, is_fallback=True),
        CheckResult("refiner", "c", "tools", False, "", 0),
        CheckResult("embeddings", "d", "embeddings", False, "", 0),
    ]
    assert roles_without_working_model(results) == ["refiner"]


def test_cli_devuelve_error_si_un_rol_no_tiene_modelos(monkeypatch, capsys):
    monkeypatch.setattr(check, "check_roles", lambda roles: [CheckResult("refiner", "c", "tools", False, "429", 0.1)])

    assert cli.main(["check-models", "--sin-embeddings"]) == 1
    assert "Sin ningún modelo disponible para: refiner" in capsys.readouterr().out


def test_cli_ok(monkeypatch, capsys):
    monkeypatch.setattr(check, "check_roles", lambda roles: [CheckResult("parser", "a", "structured", True, "ok", 0.1)])
    monkeypatch.setattr(check, "check_embeddings", lambda: CheckResult("embeddings", "e", "embeddings", True, "8 dimensiones", 0.1))

    assert cli.main(["check-models"]) == 0
    assert "Todos los roles tienen al menos un modelo disponible." in capsys.readouterr().out
