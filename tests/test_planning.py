from cvertidor.graph.planning import (
    DEFAULT_PRIORITY,
    LOW_COVERAGE_PRIORITY,
    build_plan,
    cv_haystack,
    detect_language,
    requirement_in_cv,
)
from cvertidor.schemas.job import JobRequirements

OFERTA_ES = "Buscamos Data Engineer con experiencia en Python y SQL para el equipo de datos del banco."
OFERTA_EN = "We are looking for a Data Engineer with experience in Python and SQL to join our data team."


def test_detecta_idioma():
    assert detect_language(OFERTA_ES) == "es"
    assert detect_language(OFERTA_EN) == "en"
    assert detect_language("Python SQL DBT") is None


def test_oferta_en_ingles_cambia_el_idioma_de_salida(sample_cv):
    # El modelo dijo 'es', pero el texto está en inglés: manda la detección
    job = JobRequirements(title="Data Engineer", must_have=["Python", "SQL"], language="es")
    plan = build_plan(sample_cv, job, OFERTA_EN)

    assert plan.output_language == "en"
    assert plan.cv_language == "es"
    assert plan.translate
    assert "Idioma de salida: inglés." in plan.instructions
    assert "Traduce el contenido del candidato del español al inglés." in plan.instructions


def test_oferta_en_espanol_no_traduce(sample_cv):
    plan = build_plan(sample_cv, JobRequirements(must_have=["Python"]), OFERTA_ES)
    assert plan.output_language == "es"
    assert not plan.translate
    assert "Traduce" not in plan.instructions


def test_cobertura_alta_prioriza_experiencia(sample_cv):
    job = JobRequirements(title="Data Engineer", must_have=["Python", "SQL", "DBT", "Kubernetes"])
    plan = build_plan(sample_cv, job, OFERTA_ES)

    assert plan.skill_coverage == 0.75
    assert plan.matched_requirements == ["Python", "SQL", "DBT"]
    assert plan.missing_requirements == ["Kubernetes"]
    assert plan.section_priority == DEFAULT_PRIORITY
    assert "no los agregues" in plan.instructions and "Kubernetes" in plan.instructions


def test_cobertura_baja_destaca_habilidades_y_formacion(sample_cv):
    job = JobRequirements(title="Frontend Developer", must_have=["React", "TypeScript", "CSS", "Python"])
    plan = build_plan(sample_cv, job, OFERTA_ES)

    assert plan.skill_coverage == 0.25
    assert plan.section_priority == LOW_COVERAGE_PRIORITY
    assert "habilidades transferibles" in plan.instructions


def test_consultas_de_recuperacion(sample_cv):
    job = JobRequirements(title="Data Engineer", must_have=["Python", "DBT"])
    plan = build_plan(sample_cv, job, OFERTA_ES)

    assert plan.retrieval_queries == [OFERTA_ES, "Data Engineer: experiencia con Python, DBT", "Data Engineer"]
    # Sin requisitos extraídos: solo la oferta completa
    assert build_plan(sample_cv, JobRequirements(), OFERTA_ES).retrieval_queries == [OFERTA_ES]
    assert build_plan(sample_cv, JobRequirements(), OFERTA_ES).skill_coverage is None


def test_coincidencia_de_requisitos(sample_cv):
    haystack = cv_haystack(sample_cv)
    assert requirement_in_cv("pipelines ELT", haystack)       # todas las palabras aparecen
    assert requirement_in_cv("ingeniería civil", haystack)    # sin importar tildes
    assert not requirement_in_cv("SQ", haystack)              # no coincide dentro de otra palabra
    assert not requirement_in_cv("Kubernetes", haystack)


def test_requisitos_tolerantes_a_formatos_del_modelo():
    job = JobRequirements.model_validate({"title": "none", "must_have": "Python", "language": "English"})
    assert job.title is None
    assert job.must_have == ["Python"]
    assert job.language == "en"
