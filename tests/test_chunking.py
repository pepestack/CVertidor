from langchain_core.documents import Document

from cvertidor.rag.chunking import create_cv_documents, make_candidate_id, make_doc_id


def test_make_candidate_id_quita_tildes_y_simbolos():
    assert make_candidate_id("Sebastián Alejandro Navarro Paredes") == "sebastian_alejandro_navarro_paredes"
    assert make_candidate_id("  José-María  O'Higgins ") == "jose_maria_o_higgins"


def test_make_candidate_id_vacio_usa_valor_por_defecto():
    assert make_candidate_id("¡!") == "candidato"


def test_make_doc_id_es_determinista_y_depende_de_la_coleccion():
    doc = Document(page_content="texto", metadata={"candidate_id": "ana", "section": "education"})
    assert make_doc_id("coleccion_a", doc) == make_doc_id("coleccion_a", doc)
    assert make_doc_id("coleccion_a", doc) != make_doc_id("coleccion_b", doc)


def test_un_chunk_por_responsabilidad_con_encabezado_del_cargo(sample_cv):
    docs = create_cv_documents(sample_cv)
    work = [d for d in docs if d.metadata["section"] == "work_history"]
    first_role = [d for d in work if d.metadata["role"].startswith("Data Engineer")]

    assert len(first_role) == 2
    assert [d.metadata["sub_chunk_index"] for d in first_role] == [0, 1]
    for doc in first_role:
        assert doc.page_content.startswith("Candidato: Sebastián Alejandro Navarro Paredes\nRol: Data Engineer")
        assert "\nResponsabilidades:\n- " in doc.page_content


def test_cargo_sin_responsabilidades_conserva_solo_el_encabezado(sample_cv):
    docs = create_cv_documents(sample_cv)
    junior = [d for d in docs if d.metadata.get("role") == "Analista Junior de Procesos & Métricas"]

    assert len(junior) == 1
    assert "Responsabilidades" not in junior[0].page_content
    assert "Duración: N/A" in junior[0].page_content


def test_secciones_y_metadatos(sample_cv):
    docs = create_cv_documents(sample_cv)
    sections = [d.metadata["section"] for d in docs]

    assert sections.count("education") == 1
    assert sections.count("technical_skills") == 1
    assert sections.count("certifications") == 1
    assert sections.count("languages") == 1
    assert sections.count("interests") == 1
    assert {d.metadata["candidate_id"] for d in docs} == {"sebastian_alejandro_navarro_paredes"}

    certs = next(d for d in docs if d.metadata["section"] == "certifications")
    assert "dbt Fundamentals (N/A)" in certs.page_content


def test_secciones_opcionales_vacias_no_generan_chunks(sample_cv):
    sample_cv.technical_skills = None
    sample_cv.certifications = None
    sample_cv.languages = None
    sample_cv.interests = None
    sections = {d.metadata["section"] for d in create_cv_documents(sample_cv)}

    assert sections == {"work_history", "education"}
