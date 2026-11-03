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


def test_solo_gemini_no_consulta_el_catalogo_de_openrouter(make_settings, monkeypatch):
    def no_catalog():
        raise AssertionError("no debió consultarse el catálogo de OpenRouter")

    monkeypatch.setattr(check, "openrouter_capabilities", no_catalog)
    monkeypatch.setattr(check, "_check_structured", lambda spec, role, method, settings: "ok")
    monkeypatch.setattr(check, "_check_tools", lambda spec, role, settings: "ok")

    results = check.check_roles(settings=make_settings())

    assert {r.model for r in results} == {"google:gemini-3.1-flash-lite", "google:gemini-3-flash-preview"}
    assert all(r.ok for r in results)


class _GoogleClientError(Exception):
    """Imita google.genai.errors.ClientError (usa `code`, no `status_code`)"""
    code = 429


def test_codigo_de_error_de_google_envuelto_por_langchain():
    try:
        try:
            raise _GoogleClientError("429 RESOURCE_EXHAUSTED. Quota exceeded")
        except _GoogleClientError as inner:
            raise RuntimeError("Error calling model 'gemini-3.1-flash-lite'") from inner
    except RuntimeError as wrapped:
        error = wrapped

    assert check.error_status(error) == 429
    assert check.error_status(RuntimeError("429 RESOURCE_EXHAUSTED")) == 429
    assert check.error_status(ValueError("x")) is None
    assert describe_error(error) == "límite de uso alcanzado (429), reintente más tarde"


def test_cli_cuota_de_gemini_agotada(monkeypatch, capsys, tmp_path):
    from cvertidor import pipeline

    (tmp_path / "cv.txt").write_text("x" * 60, encoding="utf-8")
    (tmp_path / "job.txt").write_text("y" * 40, encoding="utf-8")

    def raise_429(*args, **kwargs):
        raise RuntimeError("Error calling model: 429 RESOURCE_EXHAUSTED") from _GoogleClientError()

    monkeypatch.setattr(pipeline, "run_pipeline", raise_429)
    code = cli.main(["run", "--cv", str(tmp_path / "cv.txt"), "--job", str(tmp_path / "job.txt")])

    assert code == 3
    assert "cuota de la API de Gemini" in capsys.readouterr().out


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
