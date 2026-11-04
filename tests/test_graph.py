import json
from contextlib import nullcontext

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from cvertidor import cli, pipeline
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
    calls = {"retriever": [], "generate_plan": None, "parse": 0, "job": 0}
    retrieval_results = [[*WORK, SKILLS]]

    monkeypatch.setattr(runner, "ensure_db_available", lambda url: None)
    # Memoria en RAM en vez de Postgres; misma interfaz y mismo serializador del estado
    saver, store = InMemorySaver(), InMemoryStore()
    monkeypatch.setattr(runner, "open_checkpointer", lambda url: nullcontext(saver))
    monkeypatch.setattr(runner, "open_store", lambda url, info: nullcontext(store))

    def fake_parse(text, settings):
        calls["parse"] += 1
        return sample_cv.model_copy(deep=True), [], ai()

    monkeypatch.setattr(nodes, "parse_cv", fake_parse)
    monkeypatch.setattr(nodes, "sync_candidate_docs", lambda *args: SyncStats(27, 0, 0))

    job_result = {"value": JobRequirements(title="Data Engineer", must_have=["Python", "SQL"])}

    def fake_structured(role, schema, **kwargs):
        def run(_):
            calls["job"] += 1
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

    def run(job_text=JOB_ES, cv_text=CV_TEXT, **kwargs):
        # Un vector store cualquiera: el recuperador y la sincronización son falsos
        return runner.run_agent(cv_text, job_text, settings, embeddings_info=info, **kwargs)

    return {"run": run, "calls": calls, "retrieval_results": retrieval_results,
            "job_result": job_result, "settings": settings, "store": store}


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
        ["load_memory", "ingest_cv", "index_cv", "analyze_job", "plan", "retrieve", "generate", "render_and_save"])
    assert {e["run_id"] for e in events} == {result.run_id}
    assert all(e["status"] == "ok" for e in events)

    text = log.read_text(encoding="utf-8")
    for personal in (sample_cv.candidate_name, sample_cv.email, sample_cv.phone, result.candidate_id):
        assert personal not in text


def test_memoria_evita_llamadas_a_modelos_y_el_hilo_conserva_su_estado(fake_world):
    calls = fake_world["calls"]
    first = fake_world["run"](thread_id="demo")
    assert (first.cv_from_memory, first.job_from_memory, first.runs_in_thread) == (False, False, 1)
    assert (calls["parse"], calls["job"]) == (1, 1)

    # Mismo CV, misma oferta y mismo hilo: sin llamadas de extracción, el estado continúa
    # y las métricas son solo de esta ejecución (el estado se recarga desde el checkpoint)
    second = fake_world["run"](thread_id="demo")
    assert (second.cv_from_memory, second.job_from_memory, second.runs_in_thread) == (True, True, 2)
    assert (calls["parse"], calls["job"]) == (1, 1)
    assert second.parsed_cv == first.parsed_cv and second.job_reqs == first.job_reqs
    assert [m.step for m in second.metrics].count("generate") == 1
    assert len(second.similar_generations) == 1

    # Un CV editado se vuelve a procesar, y luego el anterior ya no cuenta como vigente
    assert not fake_world["run"](cv_text=CV_TEXT + " Editado.").cv_from_memory
    assert calls["parse"] == 2
    assert not fake_world["run"]().cv_from_memory
    assert calls["parse"] == 3

    # --reprocesar ignora la memoria; las preferencias guardadas llegan al plan
    fake_world["store"].put(("preferences", first.candidate_id), "p1", {"text": "omitir intereses"})
    result = fake_world["run"](reprocess=True)
    assert calls["parse"] == 4 and calls["job"] == 2
    assert result.preferences == ["omitir intereses"]
    assert "omitir intereses" in calls["generate_plan"]


def test_ningun_dato_personal_llega_a_los_modelos(fake_world, monkeypatch, sample_cv):
    cv = sample_cv.model_copy(deep=True)
    private = [cv.email, cv.phone, cv.linkedin_url, cv.candidate_name]
    cv.work_history[0].responsibilities.append(f"Coordinó con {cv.candidate_name}: {cv.email}, {cv.phone}, {cv.linkedin_url}")
    monkeypatch.setattr(nodes, "parse_cv", lambda text, settings: (cv.model_copy(deep=True), [], ai()))

    indexed = []
    monkeypatch.setattr(nodes, "sync_candidate_docs", lambda store, url, cid, docs: indexed.extend(docs) or SyncStats(1, 0, 0))

    # El generador real, con un modelo falso que copia un marcador en su respuesta
    prompts = []
    email_token = next(t for t, v in pipeline.mask_cv(cv)[1].items() if v == cv.email)

    def fake_model(*args, **kwargs):
        def run(prompt):
            prompts.append(prompt.to_string())
            adapted = cv.model_copy(update={"professional_summary": f"Contacto: {email_token}"})
            return {"parsed": adapted, "raw": ai(), "parsing_error": None}
        return RunnableLambda(run)

    monkeypatch.setattr(pipeline, "get_structured_model", fake_model)
    monkeypatch.setattr(nodes, "generate_adapted_cv", pipeline.generate_adapted_cv)

    result = fake_world["run"](JOB_ES + " Postular a reclutadora@banco.cl o al +56 9 8765 4321.")

    sent = [*(d.page_content for d in indexed), *(q for q, _ in fake_world["calls"]["retriever"]), *prompts]
    assert indexed and prompts
    for value in [*private, "reclutadora@banco.cl", "+56 9 8765 4321"]:
        assert not any(value in text for text in sent), value
    # Los marcadores de la oferta no se pueden confundir con los del CV
    assert "[DATO_OCULTO]" in prompts[0]
    # El CV final tiene los datos reales, incluido el marcador que copió el modelo
    assert result.adapted_cv.professional_summary == f"Contacto: {cv.email}"
    assert result.adapted_cv.candidate_name == cv.candidate_name
    assert all(d.metadata["candidate_id"] == result.candidate_id for d in indexed)


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
    for node in ("load_memory", "ingest_cv", "index_cv", "analyze_job", "plan", "retrieve", "generate", "render_and_save"):
        assert node in mermaid
    # Arista condicional: reintento de la recuperación o paso a la generación
    assert "retrieve -.-> retrieve" in mermaid
    assert "retrieve -.-> generate" in mermaid
    assert "load_memory -.-> analyze_job" in mermaid and "load_memory -.-> index_cv" in mermaid


def test_cli_graph_guarda_markdown(tmp_path, capsys):
    path = tmp_path / "docs" / "grafo.md"
    assert cli.main(["graph", "--salida", str(path)]) == 0
    content = path.read_text(encoding="utf-8")
    assert content.startswith("```mermaid\n") and "ingest_cv" in content
