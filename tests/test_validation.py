import httpx
import pytest

from cvertidor.llm.decisions import DecisionError, evaluate_draft
from cvertidor.schemas.cv import Certification, WorkExperience
from cvertidor.validation.checks import run_checks, status_from_scores


def test_chequeos_detectan_datos_fuera_del_cv_original(sample_cv):
    fiel = sample_cv.model_copy(deep=True)
    assert run_checks(fiel, sample_cv, "es") == []

    malo = sample_cv.model_copy(deep=True)
    malo.work_history[0].duration = "Marzo 2018 – Presente"
    malo.work_history[0].responsibilities.append("Redujo los costos en un 40% con un equipo de 20 personas.")
    malo.work_history.append(WorkExperience(company="Empresa Inventada", role="CTO"))
    malo.technical_skills = ["Python", "Kubernetes"]
    malo.certifications = [Certification(name="AWS Solutions Architect")]
    malo.interests = ["[EMAIL_1]"]
    issues = " | ".join(run_checks(malo, sample_cv, "es"))

    for expected in ("Empresa Inventada", "fechas", "Kubernetes", "AWS Solutions Architect", "'40'", "'20'", "marcadores"):
        assert expected in issues

    largo = sample_cv.model_copy(deep=True)
    largo.work_history[0].responsibilities = [f"Tarea {letra}" for letra in "abcde"]
    assert "máximo es 4" in " ".join(run_checks(largo, sample_cv, "es"))


def test_traduccion_de_habilidades_y_cifras_equivalentes(sample_cv):
    source = sample_cv.model_copy(deep=True)
    source.work_history[0].responsibilities.append("Disponibilidad de 99,9% en Excel Avanzado.")
    source.technical_skills = ["Python", "Excel Avanzado"]
    draft = source.model_copy(deep=True)
    draft.technical_skills = ["Python", "Advanced Excel"]
    draft.work_history[0].responsibilities = ["Achieved 99.9% availability in the data platform."]
    draft.professional_summary = "Data engineer with experience in the design of pipelines for the financial sector and the team."

    assert run_checks(draft, source, "en", translate=True) == []
    assert any("Advanced Excel" in i for i in run_checks(draft, source, "en", translate=False))
    assert any("idioma de salida" in i for i in run_checks(draft, source, "es"))


@pytest.mark.parametrize(
    ("scores", "status"),
    [
        ({"inventa_datos": 0.025, "omite_requisitos": 0.003}, "approved"),
        ({"inventa_datos": 0.96, "omite_requisitos": 0.0}, "rejected"),
        ({"inventa_datos": 0.0, "omite_requisitos": 0.99}, "rejected"),
        ({"inventa_datos": 0.4, "omite_requisitos": 0.0}, "uncertain"),
        ({"inventa_datos": 0.0, "omite_requisitos": 0.5}, "uncertain"),
    ],
)
def test_umbrales(scores, status):
    assert status_from_scores(scores) == status


def _transport(status: int, payload: dict, seen: list | None = None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(status, json=payload)
    return httpx.MockTransport(handler)


def test_cliente_de_decisiones(make_settings):
    answers = {name: {"type": "noul", "noul": 0.1} for name in ("inventa_datos", "omite_requisitos")}
    seen = []
    result = evaluate_draft("contexto", ["Python"], "cv", make_settings(),
                            transport=_transport(200, {"model": "respan/x", "answers": answers}, seen))
    assert result.scores == {"inventa_datos": 0.1, "omite_requisitos": 0.1}
    assert result.model == "respan/x"
    # Respan exige el estado como conversación
    assert b'"output":{"role":"assistant","content":"cv"}' in seen[0].content.replace(b": ", b":").replace(b", ", b",")

    with pytest.raises(DecisionError):
        evaluate_draft("c", [], "cv", make_settings(), transport=_transport(429, {"error": "cupo"}))
    with pytest.raises(DecisionError, match="inesperada"):
        evaluate_draft("c", [], "cv", make_settings(), transport=_transport(200, {"answers": {"inventa_datos": {"type": "score"}}}))
    with pytest.raises(DecisionError, match="OPENROUTER_API_KEY"):
        evaluate_draft("c", [], "cv", make_settings(openrouter_api_key=None))
