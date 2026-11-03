from cvertidor.render import cv_to_html, cv_to_markdown
from cvertidor.render.utils import _join, _url, esc


def test_utilidades():
    assert _url("linkedin.com/in/ana") == "https://linkedin.com/in/ana"
    assert _url("HTTP://ejemplo.cl") == "HTTP://ejemplo.cl"
    assert _join("a", None, "", "b") == "a | b"
    assert esc("<b>&</b>") == "&lt;b&gt;&amp;&lt;/b&gt;"
    assert esc(None) == ""


def test_markdown_secciones_y_enlaces(sample_cv):
    md = cv_to_markdown(sample_cv)

    assert md.startswith("# Sebastián Alejandro Navarro Paredes\n")
    assert "[linkedin.com/in/snavarro-data-engineer](https://linkedin.com/in/snavarro-data-engineer)" in md
    assert "### Data Engineer & Controller de Procesos de Datos - Grupo Financiero e Inversiones del Pacífico S.A. (Marzo 2022 – Presente)" in md
    assert "### Analista Junior de Procesos & Métricas - Consultores de Gestión y Operaciones Ltda.\n" in md
    assert "- dbt Fundamentals\n" in md
    for heading in ["## Perfil Profesional", "## Experiencia Laboral", "## Educación", "## Habilidades Técnicas",
                    "## Certificaciones", "## Idiomas", "## Intereses"]:
        assert heading in md
    assert "None" not in md


def test_markdown_omite_secciones_vacias(sample_cv):
    sample_cv.interests = None
    sample_cv.professional_summary = None
    md = cv_to_markdown(sample_cv)

    assert "## Intereses" not in md
    assert "## Perfil Profesional" not in md


def test_html_escapa_contenido(sample_cv):
    sample_cv.work_history[0].responsibilities = ["<script>alert(1)</script>"]
    html = cv_to_html(sample_cv)

    assert "<script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert '<a href="https://linkedin.com/in/snavarro-data-engineer">LinkedIn</a>' in html
    assert html.startswith("<!DOCTYPE html>") and html.endswith("</html>")
    assert "None" not in html
