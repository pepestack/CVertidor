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
    )

    db_url: str | None = Field(default=None, validation_alias="DB_URL")
    openrouter_api_key: SecretStr | None = Field(default=None, validation_alias="OPENROUTER_API_KEY")
    # API_KEY se mantiene por compatibilidad con el .env original del notebook
    google_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("GOOGLE_API_KEY", "API_KEY"),
    )

    output_dir: Path = Field(default=Path("output"), validation_alias="OUTPUT_DIR")

    def require_db_url(self) -> str:
        if not self.db_url:
            raise ValueError("Falta DB_URL en las variables de entorno.")
        return self.db_url


@lru_cache
def get_settings() -> Settings:
    cargar_secretos_colab()
    return Settings()
