import pytest

from cvertidor.config import Settings


@pytest.fixture
def entorno_limpio(monkeypatch, tmp_path):
    for name in ["DB_URL", "OPENROUTER_API_KEY", "GOOGLE_API_KEY", "API_KEY", "OUTPUT_DIR"]:
        monkeypatch.delenv(name, raising=False)
    # Evita leer el .env real del proyecto
    monkeypatch.chdir(tmp_path)


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
