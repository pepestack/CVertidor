"""Configuración del proyecto: variables de entorno (.env local) o secretos de Google Colab."""

import os
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Secretos que se copian desde Colab (userdata) al entorno, si se ejecuta en Colab
COLAB_SECRETS = ("DB_URL", "OPENROUTER_API_KEY", "GOOGLE_API_KEY")

# Respaldo de los roles de chat: Flash Lite responde 503 (alta demanda) en horas punta
GEMINI_FALLBACK = "google:gemini-3-flash-preview"


def cargar_secretos_colab() -> bool:
    """Copia los secretos de Colab a os.environ; no hace nada fuera de Colab"""
    try:
        from google.colab import userdata
    except ImportError:
        return False

    for name in COLAB_SECRETS:
        if os.getenv(name):
            continue
        try:
            value = userdata.get(name)
        except Exception:
            # El secreto no existe o no tiene acceso concedido en el cuaderno
            continue
        if value:
            os.environ[name] = value
    return True


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Permite Settings(openrouter_api_key=...) además del nombre de la variable de entorno
        populate_by_name=True,
    )

    db_url: str | None = Field(default=None, validation_alias="DB_URL")
    # Proveedor principal. API_KEY se mantiene por compatibilidad con el .env original del notebook
    google_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("GOOGLE_API_KEY", "API_KEY"),
    )
    # Opcional: solo si algún rol usa modelos de OpenRouter (openrouter:<modelo>)
    openrouter_api_key: SecretStr | None = Field(default=None, validation_alias="OPENROUTER_API_KEY")

    output_dir: Path = Field(default=Path("output"), validation_alias="OUTPUT_DIR")

    # Modelos por rol con formato "<proveedor>:<modelo>" (proveedor: google | openrouter).
    # Sin prefijo se asume openrouter. Los fallbacks se separan por comas.
    # Por defecto todos los roles usan Gemini, el mismo modelo del notebook; el perfil
    # gratuito de OpenRouter está documentado en .env.example
    parser_model: str = "google:gemini-3.1-flash-lite"
    parser_model_fallbacks: str = GEMINI_FALLBACK
    parser_structured_method: str | None = None

    # Vacío = mismos modelos que el parser
    job_model: str | None = None
    job_model_fallbacks: str | None = None
    job_structured_method: str | None = None

    generator_model: str = "google:gemini-3.1-flash-lite"
    generator_model_fallbacks: str = GEMINI_FALLBACK
    generator_structured_method: str | None = None

    refiner_model: str = "google:gemini-3.1-flash-lite"
    refiner_model_fallbacks: str = GEMINI_FALLBACK
    refiner_structured_method: str | None = None

    escalation_model: str = "google:gemini-3.1-flash-lite"
    escalation_model_fallbacks: str = GEMINI_FALLBACK
    escalation_structured_method: str | None = None

    embedding_model: str = "google:models/gemini-embedding-2"

    # Reintentos con backoff exponencial ante 429/5xx y tiempo máximo por llamada (segundos).
    # Solo el último modelo de cada cadena reintenta; los anteriores ceden de inmediato al respaldo
    llm_max_retries: int = 3
    llm_timeout: float = 120.0

    def require_db_url(self) -> str:
        if not self.db_url:
            raise ValueError("Falta DB_URL en las variables de entorno.")
        return self.db_url

    def role_models(self, role: str) -> tuple[list[str], str | None]:
        """Modelos (principal + fallbacks) y método de salida estructurada forzado de un rol"""
        if role not in ROLES:
            raise ValueError(f"Rol desconocido: '{role}'. Roles válidos: {', '.join(ROLES)}")

        primary = getattr(self, f"{role}_model")
        fallbacks = getattr(self, f"{role}_model_fallbacks")
        method = getattr(self, f"{role}_structured_method")

        # El analizador de ofertas hereda la configuración del parser si no tiene la suya
        if role == "job" and not primary:
            primary, fallbacks = self.parser_model, self.parser_model_fallbacks
            method = method or self.parser_structured_method

        models = [primary] + [m.strip() for m in (fallbacks or "").split(",") if m.strip()]
        return models, method or None


# Roles de modelo de lenguaje del agente
ROLES = ("parser", "job", "generator", "refiner", "escalation")


@lru_cache
def get_settings() -> Settings:
    cargar_secretos_colab()
    return Settings()
