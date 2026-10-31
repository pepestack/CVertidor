"""Configuración del proyecto: variables de entorno (.env local) o secretos de Google Colab."""

import os
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Secretos que se copian desde Colab (userdata) al entorno, si se ejecuta en Colab
COLAB_SECRETS = ("DB_URL", "OPENROUTER_API_KEY", "GOOGLE_API_KEY")


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
    openrouter_api_key: SecretStr | None = Field(default=None, validation_alias="OPENROUTER_API_KEY")
    # API_KEY se mantiene por compatibilidad con el .env original del notebook
    google_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("GOOGLE_API_KEY", "API_KEY"),
    )

    output_dir: Path = Field(default=Path("output"), validation_alias="OUTPUT_DIR")

    # Modelos por rol con formato "<proveedor>:<modelo>" (proveedor: openrouter | google).
    # Sin prefijo se asume openrouter. Los fallbacks se separan por comas.
    parser_model: str = "openrouter:nvidia/nemotron-3.5-super-vl:free"
    parser_model_fallbacks: str = "openrouter:google/gemma-4-31b-it:free,openrouter:nvidia/nemotron-3-ultra-550b-a55b:free"
    parser_structured_method: str | None = None

    # Vacío = mismos modelos que el parser
    job_model: str | None = None
    job_model_fallbacks: str | None = None
    job_structured_method: str | None = None

    # Gemma escribe bien en español, pero su pool gratuito (Google AI Studio) se satura
    # seguido (429 upstream), por eso va como respaldo y no como principal
    generator_model: str = "openrouter:nvidia/nemotron-3.5-super-vl:free"
    generator_model_fallbacks: str = "openrouter:google/gemma-4-31b-it:free,openrouter:nvidia/nemotron-3-ultra-550b-a55b:free"
    generator_structured_method: str | None = None

    refiner_model: str = "openrouter:nvidia/nemotron-3-ultra-550b-a55b:free"
    refiner_model_fallbacks: str = "openrouter:google/gemma-4-31b-it:free"
    refiner_structured_method: str | None = None

    escalation_model: str = "openrouter:nvidia/nemotron-3-ultra-550b-a55b:free"
    escalation_model_fallbacks: str = ""
    escalation_structured_method: str | None = None

    embedding_model: str = "openrouter:nvidia/nemotron-3.5-embed-1b:free"

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
