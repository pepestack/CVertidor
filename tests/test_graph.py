import json
from contextlib import nullcontext

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from cvertidor import cli, pipeline
from cvertidor.graph import builder, edges, nodes, refine, runner
from cvertidor.llm.decisions import DecisionError, DecisionResult
from cvertidor.llm.embeddings import EmbeddingsInfo
from cvertidor.pipeline import GenerationError, InputError
from cvertidor.rag.vector_store import SyncStats
from cvertidor.schemas.evaluation import JudgeReview
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
    calls = {"retriever": [], "generate_plan": None, "parse": 0, "job": 0, "judge": 0, "generate": 0, "reviewed": []}
    decision = {"value": DecisionResult({"inventa_datos": 0.02, "omite_requisitos": 0.01}, "fake-decision")}

    def fake_decision(context, requirements, draft, settings):
        calls["reviewed"] += [context, draft]
        if isinstance(decision["value"], Exception):
            raise decision["value"]
        return decision["value"]

    monkeypatch.setattr(nodes, "evaluate_draft", fake_decision)
    judge_review = {"value": JudgeReview(approved=True)}
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
            calls["judge" if role == "escalation" else "job"] += 1
            value = (judge_review if role == "escalation" else job_result)["value"]
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

    def fake_generate(context, job_text, parsed_cv, settings, plan="", current_draft=None, human_edit=False):
        calls["current_draft"] = current_draft
        calls.setdefault("drafts", []).append((current_draft, human_edit))
        calls["generate_plan"] = plan
        calls["generate"] += 1
        calls["context"] = context
        draft = parsed_cv.model_copy(deep=True)
        # Cada intento cambia el borrador, salvo que el test pida lo contrario
        if not calls.get("same_draft"):
            draft.interests = [f"Intento {calls['generate']}"]
        return draft, ai()

    monkeypatch.setattr(nodes, "generate_adapted_cv", fake_generate)

    settings = make_settings(db_url="postgresql://falsa", output_dir=tmp_path / "out", log_dir=tmp_path / "logs")
    info = EmbeddingsInfo(DeterministicFakeEmbedding(size=8), "google", "google:fake")

    def run(job_text=JOB_ES, cv_text=CV_TEXT, **kwargs):
        # Un vector store cualquiera: el recuperador y la sincronización son falsos
        kwargs.setdefault("auto_approve", True)
        return runner.run_agent(cv_text, job_text, settings, embeddings_info=info, **kwargs)

    def resume(thread_id, action, feedback=None):
        # El grafo se vuelve a compilar: equivale a retomar desde otro proceso
        return runner.resume_agent(thread_id, action, feedback, settings, embeddings_info=info)

    return {"run": run, "resume": resume, "calls": calls, "retrieval_results": retrieval_results,
            "job_result": job_result, "settings": settings, "store": store, "info": info,
            "decision": decision, "judge_review": judge_review}


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
    record = fake_world["store"].search(("generations", result.candidate_id))[0].value
    assert record["generation_attempts"] == 1


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
    assert next(e for e in result.audit_events if e.get("decision") == "generar_sin_experiencia")


def test_oferta_en_ingles_genera_en_ingles(fake_world):
    # El generador falso responde en español: el chequeo de idioma lo detecta y decide una persona
    pending = fake_world["run"](JOB_EN, thread_id="ingles")
    assert any("idioma de salida es 'en'" in issue for issue in pending.issues)
    result = fake_world["resume"]("ingles", "aceptar")

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
        ["load_memory", "ingest_cv", "index_cv", "analyze_job", "plan", "retrieve", "generate", "validate", "render_and_save"])
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

    sent = [*(d.page_content for d in indexed), *(q for q, _ in fake_world["calls"]["retriever"]), *prompts,
            *fake_world["calls"]["reviewed"]]
    assert indexed and prompts
    for value in [*private, "reclutadora@banco.cl", "+56 9 8765 4321"]:
        assert not any(value in text for text in sent), value
    # Los marcadores de la oferta no se pueden confundir con los del CV
    assert "[DATO_OCULTO]" in prompts[0]
    # El CV final tiene los datos reales, incluido el marcador que copió el modelo
    assert result.adapted_cv.professional_summary == f"Contacto: {cv.email}"
    assert result.adapted_cv.candidate_name == cv.candidate_name
    assert all(d.metadata["candidate_id"] == result.candidate_id for d in indexed)


def test_rechazo_corrige_con_retroalimentacion_hasta_tres_intentos(fake_world):
    fake_world["decision"]["value"] = DecisionResult({"inventa_datos": 0.96, "omite_requisitos": 0.01})

    pending = fake_world["run"]()

    assert fake_world["calls"]["generate"] == pending.attempts == 3
    assert "Correcciones obligatorias" in fake_world["calls"]["generate_plan"]
    assert (pending.kind, pending.status) == ("escalation", "rejected")
    assert fake_world["calls"]["judge"] == 0
    # El primer intento escribe desde cero; los reintentos editan el borrador rechazado
    first, *retries = fake_world["calls"]["drafts"]
    assert first == (None, False)
    assert len(retries) == 2
    assert all("## Experiencia Laboral" in draft and not human_edit for draft, human_edit in retries)

    # Un reintento que devuelve el mismo borrador corta el bucle: repetirlo solo gastaría tokens
    fake_world["calls"].update(generate=0, same_draft=True)
    assert fake_world["run"]().attempts == 2


def test_borrador_conserva_los_marcadores_del_cv_original():
    from cvertidor.ingestion.pii import unmask
    from cvertidor.schemas.cv import StandardCV, WorkExperience

    url_a, url_b = "https://a.example.com/x", "https://b.example.com/y"
    def cv(*urls):
        return StandardCV(candidate_name="Ana Pérez", work_history=[
            WorkExperience(company="C", role="R", responsibilities=[f"Proyecto {u}" for u in urls])])

    source, draft = cv(url_a, url_b), cv(url_b)  # el borrador omitió la primera URL

    masked = nodes.review_markdown(draft, "es", source)

    assert url_b not in masked
    restored = unmask(masked, pipeline.mask_cv(source)[1])
    assert url_b in restored and url_a not in restored


def test_seccion_omitida_se_restaura_salvo_que_la_pida_una_persona(fake_world, monkeypatch):
    def without_education(context, job_text, parsed_cv, settings, plan="", current_draft=None, human_edit=False):
        return parsed_cv.model_copy(update={"education": []}, deep=True), ai()

    monkeypatch.setattr(nodes, "generate_adapted_cv", without_education)

    result = fake_world["run"]()
    assert result.generation_attempts == 1
    assert result.adapted_cv.education == result.parsed_cv.education
    assert any("la educación" in w for w in result.warnings)

    pending = fake_world["run"](auto_approve=False, thread_id="quita-educacion")
    assert "## Educación" in pending.preview
    corrected = fake_world["resume"]("quita-educacion", "corregir", "Quita la educación")
    assert "## Educación" not in corrected.preview


def test_experiencia_omitida_se_restaura_sin_gastar_otro_intento(fake_world, monkeypatch):
    def without_roles(context, job_text, parsed_cv, settings, plan="", current_draft=None, human_edit=False):
        return parsed_cv.model_copy(update={"work_history": []}, deep=True), ai()

    monkeypatch.setattr(nodes, "generate_adapted_cv", without_roles)
    source = fake_world["run"]().parsed_cv.work_history[0]
    # Lo recuperado para la oferta va primero
    fake_world["retrieval_results"][:] = [[
        Document(page_content=f"Responsabilidades:\n- {i}", metadata={
            "section": "work_history", "company": source.company, "role": source.role, "sub_chunk_index": i})
        for i in (1, 0)
    ]]
    result = fake_world["run"]()

    assert result.generation_attempts == 1
    assert [w.company for w in result.adapted_cv.work_history] == [w.company for w in result.parsed_cv.work_history]
    assert result.adapted_cv.work_history[0].responsibilities == source.responsibilities[::-1]
    assert any("omitió la experiencia" in w for w in result.warnings)


def test_corregir_no_pierde_la_experiencia_salvo_que_se_pida(fake_world, monkeypatch):
    def without_bullets(context, job_text, parsed_cv, settings, plan="", current_draft=None, human_edit=False):
        roles = [w.model_copy(update={"responsibilities": []}) for w in parsed_cv.work_history]
        return parsed_cv.model_copy(update={"work_history": roles}, deep=True), ai()

    def without_roles(context, job_text, parsed_cv, settings, plan="", current_draft=None, human_edit=False):
        return parsed_cv.model_copy(update={"work_history": []}, deep=True), ai()

    # Un cargo que llega sin responsabilidades también se completa
    monkeypatch.setattr(nodes, "generate_adapted_cv", without_bullets)
    pending = fake_world["run"](auto_approve=False, thread_id="vinetas")
    assert "Liderar el diseño" in pending.preview

    # La persona pide agregar y el modelo borra la sección: no es lo que se pidió
    monkeypatch.setattr(nodes, "generate_adapted_cv", without_roles)
    kept = fake_world["resume"]("vinetas", "corregir", "Incluye viñetas relevantes en cada cargo")
    assert "## Experiencia Laboral" in kept.preview and "Liderar el diseño" in kept.preview
    # El contexto de la corrección trae toda la experiencia, aunque no se haya recuperado
    assert "Modelos de transformación modular" in fake_world["calls"]["reviewed"][-2]

    removed = fake_world["resume"]("vinetas", "corregir", "Quita la experiencia laboral")
    assert "## Experiencia Laboral" not in removed.preview


def test_dudoso_o_sin_modelo_de_decision_decide_el_juez(fake_world):
    fake_world["decision"]["value"] = DecisionResult({"inventa_datos": 0.4, "omite_requisitos": 0.01})
    result = fake_world["run"]()
    assert fake_world["calls"]["judge"] == 1
    assert (result.verdict.status, result.verdict.source, result.generation_attempts) == ("approved", "judge", 1)

    # Sin modelo de decisión, y además el juez rechaza: se corrige con su retroalimentación
    fake_world["decision"]["value"] = DecisionError("sin clave")
    fake_world["judge_review"]["value"] = JudgeReview(approved=False, feedback="Quita la cifra inventada.")
    pending = fake_world["run"]()
    assert "Quita la cifra inventada." in fake_world["calls"]["generate_plan"]
    assert (pending.kind, pending.attempts) == ("escalation", 3)

    # Si tampoco responde el juez, sin reintentar decide una persona, incluso con --auto
    fake_world["judge_review"]["value"] = ConnectionError("caído")
    pending = fake_world["run"]()
    assert (pending.kind, pending.status, pending.attempts) == ("escalation", "uncertain", 1)


def test_revision_humana_pausa_y_se_retoma(fake_world):
    pending = fake_world["run"](auto_approve=False, thread_id="revision")
    assert isinstance(pending, runner.PendingDecision)
    assert (pending.kind, pending.options) == ("review", ["aprobar", "corregir", "descartar"])
    assert "## Experiencia Laboral" in pending.preview
    assert not list(fake_world["settings"].output_dir.glob("*"))  # aún no se guarda nada

    # Mientras espera no se puede iniciar otra ejecución ni dar una decisión inválida
    with pytest.raises(InputError, match="decisión pendiente"):
        fake_world["run"](auto_approve=False, thread_id="revision")
    with pytest.raises(InputError, match="no válida"):
        fake_world["resume"]("revision", "aceptar")
    with pytest.raises(InputError, match="qué cambiar"):
        fake_world["resume"]("revision", "corregir", "  ")

    # Corregir regenera con la instrucción (sin datos personales) y vuelve a preguntar
    again = fake_world["resume"]("revision", "corregir", "Quita los intereses de ana@empresa.cl")
    assert isinstance(again, runner.PendingDecision) and again.attempts == 2
    context = fake_world["calls"]["context"]
    assert "Quita los intereses" in context and "ana@empresa.cl" not in context

    result = fake_world["resume"]("revision", "aprobar")
    assert result.human_decision == "aprobar" and all(p.exists() for p in result.output_paths)
    with pytest.raises(InputError, match="ninguna decisión"):
        fake_world["resume"]("revision", "aprobar")

    # Descartar termina sin archivos, aunque el hilo tenga los de la ejecución anterior
    fake_world["run"](auto_approve=False, thread_id="revision")
    discarded = fake_world["resume"]("revision", "descartar")
    assert discarded.discarded and discarded.output_paths == []


def test_correcciones_se_acumulan_editan_el_borrador_y_pueden_agregar_datos(fake_world, monkeypatch):
    calls = fake_world["calls"]

    def fake_edit(context, job_text, parsed_cv, settings, plan="", current_draft=None, human_edit=False):
        calls.update(context=context, current_draft=current_draft)
        draft = parsed_cv.model_copy(deep=True)
        if "Claude Code" in context:
            draft.technical_skills = [*draft.technical_skills, "Claude Code"]
        return draft, ai()

    monkeypatch.setattr(nodes, "generate_adapted_cv", fake_edit)
    fake_world["run"](auto_approve=False, thread_id="edita")
    assert calls["current_draft"] is None

    # Un dato que no está en el CV pasa la validación porque lo confirmó una persona; quitar
    # un requisito a propósito (omisión alta) tampoco hace que el sistema lo vuelva a poner
    fake_world["decision"]["value"] = DecisionResult({"inventa_datos": 0.01, "omite_requisitos": 0.99})
    first = fake_world["resume"]("edita", "corregir", "Añadir Claude Code")
    assert first.status == "approved" and "Claude Code" in first.preview
    assert "## Experiencia Laboral" in calls["current_draft"]

    second = fake_world["resume"]("edita", "corregir", "Quitar Looker")
    assert second.corrections == ["Añadir Claude Code", "Quitar Looker"]
    assert "Añadir Claude Code" in calls["context"] and "Quitar Looker" in calls["context"]
    assert "Claude Code" in calls["current_draft"]  # edita sobre el resultado de la corrección anterior

    result = fake_world["resume"]("edita", "aprobar")
    assert "Claude Code" in result.adapted_cv.technical_skills
    assert any("indicados por una persona" in w and "Claude Code" in w for w in result.warnings)

    # Las correcciones valen para esa ejecución: no quedan en el perfil ni en la siguiente
    fake_world["decision"]["value"] = DecisionResult({"inventa_datos": 0.01, "omite_requisitos": 0.01})
    next_run = fake_world["run"](thread_id="edita")
    assert next_run.human_corrections == [] and "Claude Code" not in next_run.parsed_cv.technical_skills


def test_escalacion_permite_corregir_una_sola_vez(fake_world):
    fake_world["decision"]["value"] = DecisionResult({"inventa_datos": 0.96, "omite_requisitos": 0.01})

    pending = fake_world["run"](thread_id="esc")  # también con --auto
    assert pending.options == ["aceptar", "corregir", "descartar"]

    # Corregir reinicia los intentos automáticos; la segunda vez ya no se ofrece
    fake_world["calls"]["generate"] = 0
    pending = fake_world["resume"]("esc", "corregir", "Quita lo inventado")
    assert fake_world["calls"]["generate"] == pending.attempts == 3
    assert pending.options == ["aceptar", "descartar"]

    result = fake_world["resume"]("esc", "aceptar")
    assert result.output_paths and any("decisión humana" in w for w in result.warnings)


def _scripted(monkeypatch, replies, seen):
    """Modelo del agente con respuestas fijas; registra lo que recibe en cada llamada"""
    queue = list(replies)

    def model(*args, **kwargs):
        def run(messages):
            seen.append(messages)
            return queue.pop(0)
        return RunnableLambda(run)

    monkeypatch.setattr(refine, "get_tool_model", model)
    monkeypatch.setattr(refine, "get_chat_model", lambda *a, **k: RunnableLambda(lambda _: AIMessage("RESUMEN COMPRIMIDO")))


def _call(name, **args):
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"id-{name}", "type": "tool_call"}])


def test_agente_edita_guarda_y_recuerda_con_confirmacion(fake_world, monkeypatch):
    with pytest.raises(InputError, match="no tiene un CV"):
        runner.chat_agent("sin-cv", "hola", fake_world["settings"], embeddings_info=fake_world["info"])

    first = fake_world["run"](thread_id="chat")
    seen = []
    _scripted(monkeypatch, [
        _call("edit_cv", output_language="en"),
        _call("remember_preference", preference="siempre en inglés", output_language="en"),
        AIMessage("Listo, [CANDIDATO_1]."),
        _call("save_cv_files"),
        _call("save_cv_files"),
        AIMessage("Guardado."),
        _call("forget_preference", preference="ingles siempre"),
        AIMessage("Se conserva."),
        _call("forget_preference", preference="siempre en inglés"),
        AIMessage("Olvidada."),
    ], seen)
    # El candidato nunca es un argumento que el modelo pueda elegir
    arguments = {name for t in refine.REFINER_TOOLS for name in t.tool_call_schema.model_json_schema().get("properties", {})}
    assert arguments == {"query", "preference", "output_language"}

    def chat(message):
        return runner.chat_agent("chat", message, fake_world["settings"], embeddings_info=fake_world["info"])

    pending = chat(f"Tradúcelo al inglés y recuérdalo. Soy {first.parsed_cv.candidate_name}, ana@empresa.cl")
    assert pending.kind == "preference" and "siempre en inglés" in pending.question

    # El dato confirmado es el texto de la persona, enmascarado; el draft se editó y está en inglés
    assert first.parsed_cv.candidate_name not in fake_world["calls"]["context"]
    assert "Tradúcelo al inglés" in fake_world["calls"]["context"]
    assert fake_world["store"].search(("preferences", first.candidate_id)) == []  # aún no confirmada

    reply = fake_world["resume"]("chat", "guardar")
    assert isinstance(reply, runner.ChatReply)
    assert reply.text == f"Listo, {first.parsed_cv.candidate_name}."  # la persona lee su dato real
    assert all(first.parsed_cv.candidate_name not in str(m.content) for call in seen for m in call)

    # Guardar dos veces lo mismo deja un solo par de archivos y un solo registro
    before = len(list(fake_world["settings"].output_dir.glob("*")))
    records = len(fake_world["store"].search(("generations", first.candidate_id), limit=50))
    saved = chat("Guárdalo")
    assert [p.suffix for p in saved.output_paths] == [".md", ".html"]
    assert len(list(fake_world["settings"].output_dir.glob("*"))) == before + 2
    assert len(fake_world["store"].search(("generations", first.candidate_id), limit=50)) == records + 1

    # Otro hilo del mismo candidato recuerda la preferencia, incluso con una oferta en español
    pending = fake_world["run"](thread_id="otro-hilo", auto_approve=False)
    assert "## Work Experience" in pending.preview
    fake_world["resume"]("otro-hilo", "descartar")

    # Olvidar también se confirma: se muestra la preferencia real, aunque se haya pedido con otras palabras
    pending = chat("Olvida lo del inglés")
    assert (pending.options, pending.question) == (["olvidar", "conservar"], "¿Olvido esta preferencia?: siempre en inglés")
    assert fake_world["resume"]("chat", "conservar").text == "Se conserva."
    assert len(fake_world["store"].search(("preferences", first.candidate_id))) == 1

    chat("Olvida lo del inglés")
    assert fake_world["resume"]("chat", "olvidar").text == "Olvidada."
    assert fake_world["store"].search(("preferences", first.candidate_id)) == []
    assert "## Experiencia Laboral" in fake_world["run"](thread_id="tercer-hilo", auto_approve=False).preview


def test_hilo_largo_respeta_el_presupuesto_y_conserva_resumen_y_borrador(fake_world, monkeypatch):
    fake_world["run"](thread_id="largo")
    seen = []
    _scripted(monkeypatch, [AIMessage("respuesta " * 100) for _ in range(12)], seen)

    for turn in range(12):
        runner.chat_agent("largo", f"Pregunta {turn}: " + "detalle " * 100, fake_world["settings"],
                          embeddings_info=fake_world["info"])

    system, *history = seen[-1]
    assert count_tokens_approximately(history) <= refine.HISTORY_TOKENS
    assert "RESUMEN COMPRIMIDO" in system.content and "## Experiencia Laboral" in system.content
    assert isinstance(history[0], HumanMessage) and "Pregunta 11" in history[-1].content


def test_fallo_del_modelo_en_un_reintento_conserva_el_borrador(fake_world, monkeypatch):
    fake_world["decision"]["value"] = DecisionResult({"inventa_datos": 0.96, "omite_requisitos": 0.01})
    first_call = nodes.generate_adapted_cv

    def fails_on_retry(*args, **kwargs):
        if fake_world["calls"]["generate"] >= 1:
            raise GenerationError("salida inválida")
        return first_call(*args, **kwargs)

    monkeypatch.setattr(nodes, "generate_adapted_cv", fails_on_retry)

    pending = fake_world["run"](thread_id="reintento-falla")

    assert (pending.kind, pending.status, pending.attempts) == ("escalation", "rejected", 2)
    assert "Intento 1" in pending.preview
    assert any("No se pudo generar" in issue for issue in pending.issues)
    # El mismo borrador no se valida otra vez: el modelo de decisión se llamó una sola vez
    assert len(fake_world["calls"]["reviewed"]) == 2

    result = fake_world["resume"]("reintento-falla", "aceptar")
    assert result.adapted_cv.interests == ["Intento 1"] and result.output_paths


def test_fallo_al_aplicar_una_correccion_no_la_da_por_aplicada(fake_world, monkeypatch):
    fake_world["run"](auto_approve=False, thread_id="correccion-falla")

    def down(*args, **kwargs):
        raise RuntimeError("503 UNAVAILABLE")

    monkeypatch.setattr(nodes, "generate_adapted_cv", down)
    pending = fake_world["resume"]("correccion-falla", "corregir", "Añadir Claude Code")

    assert pending.kind == "escalation" and pending.corrections == []
    assert any("no se aplicó" in issue for issue in pending.issues)

    result = fake_world["resume"]("correccion-falla", "aceptar")
    assert not any("indicados por una persona" in w for w in result.warnings)


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
    for node in ("load_memory", "ingest_cv", "index_cv", "analyze_job", "plan", "retrieve", "generate", "validate", "judge", "human_review", "human_escalation", "render_and_save", "refine_agent", "tools", "confirm_preference"):
        assert node in mermaid
    # Arista condicional: reintento de la recuperación o paso a la generación
    assert "retrieve -.-> retrieve" in mermaid
    assert "retrieve -.-> generate" in mermaid
    assert "generate -.-> human_escalation" in mermaid
    assert "load_memory -.-> analyze_job" in mermaid and "load_memory -.-> index_cv" in mermaid
    assert "refine_agent -.-> plan" not in mermaid and "confirm_preference -.-> plan" not in mermaid


def test_cli_graph_guarda_markdown(tmp_path, capsys):
    path = tmp_path / "docs" / "grafo.md"
    assert cli.main(["graph", "--salida", str(path)]) == 0
    content = path.read_text(encoding="utf-8")
    assert content.startswith("```mermaid\n") and "ingest_cv" in content
