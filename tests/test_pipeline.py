import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from cvertidor import pipeline
from cvertidor.graph import runner
from cvertidor.ingestion.pii import mask_pii
from cvertidor.pipeline import GenerationError, InputError, restore_contact, validate_text
from cvertidor.rag.retrieval import format_docs, similarity_threshold
from cvertidor.schemas.cv import StandardCV

ORIGINAL = """Ana Pérez
Email: ana@empresa.cl | Tel: +56 9 1234 5678
LinkedIn: linkedin.com/in/ana-perez
Experiencia: Data Engineer en TechCorp (2021 - Presente)"""


def test_validate_text():
    assert validate_text("  " + "x" * 60 + " ", 50, "El CV") == "x" * 60
    with pytest.raises(InputError, match="El CV"):
        validate_text("corto", 50, "El CV")
    with pytest.raises(InputError):
        validate_text(None, 30, "La oferta")


def test_restore_contact_restaura_marcadores_y_descarta_lo_inventado():
    masked = mask_pii(ORIGINAL)
    email_token = next(t for t, v in masked.mapping.items() if v == "ana@empresa.cl")
    url_token = next(t for t, v in masked.mapping.items() if v.startswith("linkedin"))
    # Lo que devolvería un modelo: marcadores copiados y datos inventados
    parsed = StandardCV(
        candidate_name="Ana Pérez",
        email=email_token,
        phone="+56 9 XXXX XXXX",
        linkedin_url=f"https://{url_token}",
        portfolio_url="github.com/anaperez",
    )

    restored, removed = restore_contact(parsed, masked, ORIGINAL)

    assert restored.email == "ana@empresa.cl"
    assert restored.linkedin_url == "https://linkedin.com/in/ana-perez"
    assert restored.phone is None
    assert restored.portfolio_url is None
    assert len(removed) == 2


def test_contacto_correcto_sin_marcador_se_conserva():
    parsed = StandardCV(candidate_name="Ana", phone="+56912345678")
    restored, removed = restore_contact(parsed, mask_pii(ORIGINAL), ORIGINAL)
    assert restored.phone == "+56912345678"
    assert removed == []


def _fake_structured(result: dict):
    return lambda *args, **kwargs: RunnableLambda(lambda _: result)


def test_generate_copia_contacto_del_cv_original(monkeypatch, make_settings):
    adapted = StandardCV(candidate_name="Otro Nombre", email="inventado@x.cl", professional_summary="Perfil")
    monkeypatch.setattr(pipeline, "get_structured_model",
                        _fake_structured({"parsed": adapted, "raw": AIMessage(content=""), "parsing_error": None}))
    original = StandardCV(candidate_name="Ana Pérez", email="ana@empresa.cl", location="Santiago")

    result, _ = pipeline.generate_adapted_cv("contexto", "oferta", original, make_settings())

    assert result.candidate_name == "Ana Pérez"
    assert result.email == "ana@empresa.cl"
    assert result.location == "Santiago"
    assert result.professional_summary == "Perfil"


def test_generate_conserva_los_cargos_originales(monkeypatch, make_settings, sample_cv):
    from cvertidor.schemas.cv import WorkExperience

    translated = StandardCV(candidate_name="x", work_history=[
        WorkExperience(company="Grupo Financiero e Inversiones del Pacífico S.A.", role="Data Process Controller"),
        WorkExperience(company="Empresa Inventada", role="CTO"),
    ])
    monkeypatch.setattr(pipeline, "get_structured_model",
                        _fake_structured({"parsed": translated, "raw": AIMessage(content=""), "parsing_error": None}))

    result, _ = pipeline.generate_adapted_cv("contexto", "oferta", sample_cv, make_settings())

    assert result.work_history[0].role == "Data Engineer & Controller de Procesos de Datos"
    # Una empresa que no está en el CV original no se toca aquí (la validación del batch 6 la detecta)
    assert result.work_history[1].role == "CTO"


def test_reintento_de_validacion_edita_el_borrador_y_conserva_los_cargos(monkeypatch, make_settings, sample_cv):
    from cvertidor.schemas.cv import WorkExperience

    seen = {}
    translated = StandardCV(candidate_name="x", work_history=[
        WorkExperience(company="Grupo Financiero e Inversiones del Pacífico S.A.", role="Data Process Controller"),
    ])

    def fake_model(*args, **kwargs):
        def run(prompt_value):
            seen["prompt"] = prompt_value.to_string()
            return {"parsed": translated, "raw": AIMessage(content=""), "parsing_error": None}
        return RunnableLambda(run)

    monkeypatch.setattr(pipeline, "get_structured_model", fake_model)

    result, _ = pipeline.generate_adapted_cv("contexto", "oferta", sample_cv, make_settings(), current_draft="BORRADOR")

    assert "BORRADOR RECHAZADO" in seen["prompt"] and "correcciones obligatorias" in seen["prompt"]
    assert result.work_history[0].role == "Data Engineer & Controller de Procesos de Datos"


def test_generate_sin_salida_valida_da_error(monkeypatch, make_settings):
    raw = AIMessage(content="", response_metadata={"finish_reason": "length"})
    monkeypatch.setattr(pipeline, "get_structured_model",
                        _fake_structured({"parsed": None, "raw": raw, "parsing_error": "JSON incompleto"}))

    with pytest.raises(GenerationError, match="finish_reason=length"):
        pipeline.generate_adapted_cv("contexto", "oferta", StandardCV(candidate_name="Ana"), make_settings())


def test_parse_cv_no_envia_contacto_al_modelo(monkeypatch, make_settings):
    seen = {}

    def fake_model(*args, **kwargs):
        def run(prompt_value):
            seen["prompt"] = prompt_value.to_string()
            return {"parsed": StandardCV(candidate_name="Ana Pérez"), "raw": AIMessage(content=""), "parsing_error": None}
        return RunnableLambda(run)

    monkeypatch.setattr(pipeline, "get_structured_model", fake_model)
    pipeline.parse_cv(ORIGINAL, make_settings())

    assert "ana@empresa.cl" not in seen["prompt"]
    assert "+56 9 1234 5678" not in seen["prompt"]
    assert "linkedin.com/in/ana-perez" not in seen["prompt"]
    assert "Data Engineer en TechCorp" in seen["prompt"]


def test_format_docs_agrupa_responsabilidades_por_cargo():
    header = "Candidato: Ana\nRol: DE - TechCorp\nDuración: 2021"
    meta = {"section": "work_history", "company": "TechCorp", "role": "DE", "duration": "2021"}
    docs = [
        Document(page_content=f"{header}\nResponsabilidades:\n- Pipelines", metadata=meta),
        Document(page_content=f"{header}\nResponsabilidades:\n- RAG", metadata=meta),
        Document(page_content=f"{header}\nResponsabilidades:\n- RAG", metadata=meta),  # repetido
        Document(page_content="Candidato: Ana\nIdiomas: Español", metadata={"section": "languages"}),
    ]

    assert format_docs(docs) == (
        "--- Chunk (work_history) ---\nRol: DE - TechCorp\nDuración: 2021\nResponsabilidades:\n- Pipelines\n- RAG"
        "\n\n--- Chunk (languages) ---\nCandidato: Ana\nIdiomas: Español"
    )


def test_umbral_por_modelo_de_embeddings():
    assert similarity_threshold("google") == 0.70
    assert similarity_threshold("minilm") == 0.3
    assert similarity_threshold("modelo_desconocido") == 0.3


def test_base_caida_falla_antes_de_llamar_a_los_modelos(monkeypatch, make_settings):
    import psycopg

    def no_models(*args, **kwargs):
        raise AssertionError("no debió llamarse a un modelo")

    monkeypatch.setattr(pipeline, "get_structured_model", no_models)
    settings = make_settings(db_url="postgresql://u:p@127.0.0.1:1/nada")

    with pytest.raises(psycopg.OperationalError):
        runner.run_agent("x" * 60, "y" * 40, settings)


def test_cli_cupo_diario_agotado(monkeypatch, capsys, tmp_path):
    import httpx
    import openai

    from cvertidor import cli

    (tmp_path / "cv.txt").write_text("x" * 60, encoding="utf-8")
    (tmp_path / "job.txt").write_text("y" * 40, encoding="utf-8")
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    error = openai.RateLimitError("Rate limit exceeded: free-models-per-day",
                                  response=httpx.Response(429, request=request), body=None)

    def raise_429(*args, **kwargs):
        raise error

    monkeypatch.setattr(runner, "run_agent", raise_429)
    code = cli.main(["run", "--cv", str(tmp_path / "cv.txt"), "--job", str(tmp_path / "job.txt")])

    assert code == 3
    assert "cupo diario" in capsys.readouterr().out
