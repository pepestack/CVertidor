import json

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from cvertidor import cli
from cvertidor.graph import builder, edges, nodes, runner
from cvertidor.llm.embeddings import EmbeddingsInfo
from cvertidor.pipeline import GenerationError
from cvertidor.rag.vector_store import SyncStats
from cvertidor.schemas.job import JobRequirements

CV_TEXT = "Sebastián Navarro. Data Engineer con experiencia en Python, SQL y DBT en el sector financiero."
JOB_ES = "Buscamos Data Engineer con experiencia en Python y SQL para el equipo de datos del banco."
JOB_EN = "We are looking for a Data Engineer with experience in Python and SQL to join our data team."

WORK_META = {"section": "work_history", "company": "Pacífico", "role": "DE", "duration": "2022"}
WORK = [Document(page_content="Rol: DE\nResponsabilidades:\n- Pipelines con Python", metadata=WORK_META),
        Document(page_content="Rol: DE\nResponsabilidades:\n- Modelos con DBT", metadata=WORK_META)]
SKILLS = Document(page_content="Habilidades técnicas: Python, SQL", metadata={"section": "technical_skills"})


def ai(model="fake-model"):
    return AIMessage(content="", response_metadata={"model_name": model},
                     usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})


@pytest.fixture
def fake_world(monkeypatch, make_settings, sample_cv, tmp_path):
    """Grafo completo sin red ni base de datos: modelos, pgvector y recuperador falsos"""
    calls = {"retriever": [], "generate_plan": None}
    retrieval_results = [[*WORK, SKILLS]]

    monkeypatch.setattr(runner, "ensure_db_available", lambda url: None)
    monkeypatch.setattr(nodes, "parse_cv", lambda text, settings: (sample_cv.model_copy(deep=True), [], ai()))
    monkeypatch.setattr(nodes, "sync_candidate_docs", lambda *args: SyncStats(27, 0, 0))

    job_result = {"value": JobRequirements(title="Data Engineer", must_have=["Python", "SQL"])}

    def fake_structured(role, schema, **kwargs):
        def run(_):
            value = job_result["value"]
            if isinstance(value, Exception):
                raise value
            return {"parsed": value, "raw": ai(), "parsing_error": None}
        return RunnableLambda(run)

    monkeypatch.setattr(nodes, "get_structured_model", fake_structured)

    def fake_retriever(vector_store, embeddings, candidate_id, threshold):
        def run(query):
            calls["retriever"].append((query, threshold))
            index = min(len(calls["retriever"]) - 1, len(retrieval_results) - 1)
            return retrieval_results[index]
        return RunnableLambda(run)

    monkeypatch.setattr(nodes, "build_retriever", fake_retriever)

    def fake_generate(context, job_text, parsed_cv, settings, plan=""):
        calls["generate_plan"] = plan
        calls["context"] = context
        return parsed_cv.model_copy(deep=True), ai()

    monkeypatch.setattr(nodes, "generate_adapted_cv", fake_generate)

    settings = make_settings(db_url="postgresql://falsa", output_dir=tmp_path / "out", log_dir=tmp_path / "logs")
    info = EmbeddingsInfo(DeterministicFakeEmbedding(size=8), "google", "google:fake")

    def run(job_text=JOB_ES):
        # Un vector store cualquiera: el recuperador y la sincronización son falsos
        result = runner.run_agent(CV_TEXT, job_text, settings, embeddings_info=info)
        return result

    return {"run": run, "calls": calls, "retrieval_results": retrieval_results,
            "job_result": job_result, "settings": settings}


@pytest.fixture(autouse=True)
def sin_pgvector(monkeypatch):
    from cvertidor.graph import state

    monkeypatch.setattr(state, "get_vector_store", lambda *args: object())


def test_flujo_completo_con_experiencia_relevante(fake_world):
    result = fake_world["run"]()

    assert result.retrieval_attempts == 1
    assert result.plan.output_language == "es"
    assert result.job_reqs.must_have == ["Python", "SQL"]
    assert [p.suffix for p in result.output_paths] == [".md", ".html"]
    assert all(p.exists() for p in result.output_paths)
    assert "## Experiencia Laboral" in result.markdown
    # El plan llega al prompt de generación y la educación siempre va al contexto
    assert "Idioma de salida: español." in fake_world["calls"]["generate_plan"]
    assert "--- Chunk (education) ---" in fake_world["calls"]["context"]
    assert [m.step for m in result.metrics].count("generate") == 1


def test_sin_experiencia_reintenta_con_consultas_mas_amplias(fake_world):
    fake_world["retrieval_results"][:] = [[SKILLS, WORK[0]], [], [WORK[1]]]

    result = fake_world["run"]()

    calls = fake_world["calls"]["retriever"]
    assert result.retrieval_attempts == 3
    # Cada intento usa la siguiente consulta del plan y un umbral 0.05 más bajo
    assert [q for q, _ in calls] == result.plan.retrieval_queries
    assert [t for _, t in calls] == [0.7, 0.65, 0.6]
    # Se conserva lo recuperado en intentos anteriores
    assert {d.metadata["section"] for d in result.retrieved_docs} == {"technical_skills", "work_history"}
    decisions = [e["decision"] for e in result.audit_events if e["node"] == "retrieve"]
    assert decisions == ["reintentar", "reintentar", "suficiente"]
    assert not result.warnings


def test_agota_los_intentos_y_genera_con_aviso(fake_world):
    fake_world["retrieval_results"][:] = [[SKILLS]]

    result = fake_world["run"]()

    assert result.retrieval_attempts == edges.MAX_RETRIEVAL_ATTEMPTS
    assert any("Ninguna experiencia laboral" in w for w in result.warnings)
    assert result.audit_events[-3]["decision"] == "generar_sin_experiencia"


def test_oferta_en_ingles_genera_en_ingles(fake_world):
    result = fake_world["run"](JOB_EN)

    assert result.plan.output_language == "en"
    assert "Traduce el contenido del candidato del español al inglés." in fake_world["calls"]["generate_plan"]
    assert "## Work Experience" in result.markdown
    assert "<html lang='en'>" in result.html


def test_oferta_sin_requisitos_validos_sigue_el_flujo(fake_world):
    fake_world["job_result"]["value"] = GenerationError("JSON inválido")

    result = fake_world["run"]()

    assert result.job_reqs.must_have == []
    assert result.plan.retrieval_queries == [JOB_ES]
    assert any("requisitos de la oferta" in w for w in result.warnings)
    job_event = next(e for e in result.audit_events if e["node"] == "analyze_job")
    assert job_event["decision"] == "sin_requisitos"


def test_auditoria_un_evento_por_nodo_sin_datos_personales(fake_world, sample_cv):
    result = fake_world["run"]()

    log = fake_world["settings"].log_dir / "audit.jsonl"
    events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert sorted(e["node"] for e in events) == sorted(
        ["ingest_cv", "analyze_job", "plan", "retrieve", "generate", "render_and_save"])
    assert {e["run_id"] for e in events} == {result.run_id}
    assert all(e["status"] == "ok" for e in events)

    text = log.read_text(encoding="utf-8")
    for personal in (sample_cv.candidate_name, sample_cv.email, sample_cv.phone, result.candidate_id):
        assert personal not in text


def test_error_en_un_nodo_queda_auditado(fake_world, monkeypatch):
    def broken(*args, **kwargs):
        raise GenerationError("salida inválida")

    monkeypatch.setattr(nodes, "generate_adapted_cv", broken)
    with pytest.raises(GenerationError):
        fake_world["run"]()

    log = fake_world["settings"].log_dir / "audit.jsonl"
    last = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    assert (last["node"], last["status"], last["error"]) == ("generate", "error", "GenerationError")


@pytest.mark.parametrize(
    ("docs", "attempts", "expected"),
    [(WORK, 1, "generate"), (WORK[:1], 1, "retrieve"), ([], 2, "retrieve"), ([SKILLS], 3, "generate")],
)
def test_ruta_despues_de_recuperar(docs, attempts, expected):
    assert edges.route_after_retrieve({"retrieved_docs": docs, "retrieval_attempts": attempts}) == expected


def test_diagrama_mermaid():
    mermaid = builder.graph_mermaid()
    for node in ("ingest_cv", "analyze_job", "plan", "retrieve", "generate", "render_and_save"):
        assert node in mermaid
    # Arista condicional: reintento de la recuperación o paso a la generación
    assert "retrieve -.-> retrieve" in mermaid
    assert "retrieve -.-> generate" in mermaid
    assert "__start__ --> analyze_job" in mermaid and "__start__ --> ingest_cv" in mermaid


def test_cli_graph_guarda_markdown(tmp_path, capsys):
    path = tmp_path / "docs" / "grafo.md"
    assert cli.main(["graph", "--salida", str(path)]) == 0
    content = path.read_text(encoding="utf-8")
    assert content.startswith("```mermaid\n") and "ingest_cv" in content
