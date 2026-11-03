import ipaddress
import socket

import pytest

from cvertidor.config import Settings
from cvertidor.schemas.cv import Certification, Education, StandardCV, WorkExperience


@pytest.fixture(autouse=True)
def sin_red_externa(monkeypatch):
    """Ningún test se conecta fuera de la máquina: así nunca gastan tokens ni cupo de API.
    Las conexiones locales (127.0.0.1, ::1) se permiten para las pruebas de base de datos"""
    original = socket.socket.connect

    def guarded(sock, address):
        host = address[0] if isinstance(address, tuple) else address
        try:
            local = ipaddress.ip_address(host).is_loopback
        except ValueError:
            local = host == "localhost"
        if not local:
            raise RuntimeError(f"Conexión externa bloqueada en los tests: {address}")
        return original(sock, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)


@pytest.fixture(autouse=True)
def sin_catalogo_de_openrouter(monkeypatch):
    """Ningún test consulta el catálogo de modelos por red"""
    from cvertidor.llm import capabilities, providers

    monkeypatch.setattr(capabilities, "openrouter_capabilities", lambda: {})
    monkeypatch.setattr(providers, "openrouter_capabilities", lambda: {})


@pytest.fixture
def entorno_limpio(monkeypatch, tmp_path):
    """Sin variables del sistema ni el .env real del proyecto"""
    for name in list(Settings.model_fields) + ["DB_URL", "OPENROUTER_API_KEY", "GOOGLE_API_KEY", "API_KEY", "OUTPUT_DIR"]:
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def make_settings(entorno_limpio):
    """Settings de prueba con claves falsas de Google y OpenRouter (nunca se llama a la red)"""
    def factory(**overrides) -> Settings:
        values = {"google_api_key": "g-prueba", "openrouter_api_key": "sk-or-prueba"} | overrides
        return Settings(**values)
    return factory


@pytest.fixture
def sample_cv() -> StandardCV:
    return StandardCV(
        candidate_name="Sebastián Alejandro Navarro Paredes",
        email="sebastian.navarro.dev@emailficticio.cl",
        phone="+56 9 6123 4567",
        location="Las Condes, Santiago, Chile",
        linkedin_url="linkedin.com/in/snavarro-data-engineer",
        portfolio_url="https://github.com/snavarro-data",
        professional_summary="Ingeniero Civil Industrial especializado en arquitecturas de datos.",
        education=[
            Education(institution="Universidad de Chile", degree="Ingeniería Civil Industrial", year="2012 – 2017"),
        ],
        work_history=[
            WorkExperience(
                company="Grupo Financiero e Inversiones del Pacífico S.A.",
                role="Data Engineer & Controller de Procesos de Datos",
                duration="Marzo 2022 – Presente",
                responsibilities=[
                    "Liderar el diseño de pipelines ELT/ETL con Python y SQL.",
                    "Modelos de transformación modular con DBT.",
                ],
            ),
            WorkExperience(
                company="Consultores de Gestión y Operaciones Ltda.",
                role="Analista Junior de Procesos & Métricas",
                duration=None,
                responsibilities=[],
            ),
        ],
        technical_skills=["Python", "SQL", "DBT"],
        certifications=[
            Certification(name="Google Cloud Professional Data Engineer", issuing_organization="Google Cloud", issue_date="2024"),
            Certification(name="dbt Fundamentals"),
        ],
        languages=["Español: Nativo", "Inglés: Avanzado (C1)"],
        interests=["Ajedrez competitivo", "Senderismo"],
    )
