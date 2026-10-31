import pytest

from cvertidor.config import Settings


def test_lee_variables_de_entorno(entorno_limpio, monkeypatch):
    monkeypatch.setenv("DB_URL", "postgresql://u:p@localhost:5432/db")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-prueba")
    settings = Settings()

    assert settings.require_db_url() == "postgresql://u:p@localhost:5432/db"
    assert settings.openrouter_api_key.get_secret_value() == "sk-or-prueba"
    # La clave no se expone al imprimir la configuración
    assert "sk-or-prueba" not in repr(settings)


def test_api_key_legada_se_acepta_como_clave_de_google(entorno_limpio, monkeypatch):
    monkeypatch.setenv("API_KEY", "clave-google")
    assert Settings().google_api_key.get_secret_value() == "clave-google"


def test_lee_archivo_env(entorno_limpio, tmp_path):
    (tmp_path / ".env").write_text("DB_URL=postgresql://desde-archivo\n", encoding="utf-8")
    assert Settings().db_url == "postgresql://desde-archivo"


def test_db_url_faltante_da_error_claro(entorno_limpio):
    with pytest.raises(ValueError, match="Falta DB_URL"):
        Settings().require_db_url()


def test_modelos_por_rol_con_fallbacks(make_settings):
    settings = make_settings(generator_model="openrouter:a/b:free", generator_model_fallbacks=" c/d:free , ,google:gemini-x ")
    models, method = settings.role_models("generator")

    assert models == ["openrouter:a/b:free", "c/d:free", "google:gemini-x"]
    assert method is None


def test_job_hereda_modelos_del_parser(make_settings):
    settings = make_settings(parser_model="p/1", parser_model_fallbacks="p/2", parser_structured_method="json_mode")
    assert settings.role_models("job") == (["p/1", "p/2"], "json_mode")

    propio = make_settings(job_model="j/1", job_model_fallbacks="")
    assert propio.role_models("job") == (["j/1"], None)


def test_variables_de_entorno_de_modelos(entorno_limpio, monkeypatch):
    monkeypatch.setenv("REFINER_MODEL", "openrouter:x/y")
    monkeypatch.setenv("REFINER_MODEL_FALLBACKS", "")
    monkeypatch.setenv("REFINER_STRUCTURED_METHOD", "function_calling")
    assert Settings().role_models("refiner") == (["openrouter:x/y"], "function_calling")


def test_rol_desconocido(make_settings):
    with pytest.raises(ValueError, match="Rol desconocido"):
        make_settings().role_models("traductor")
