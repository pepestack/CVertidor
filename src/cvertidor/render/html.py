"""Generador de HTML (estilo Harvard) para un StandardCV."""

from cvertidor.render.utils import _join, _url, esc
from cvertidor.schemas.cv import StandardCV


def cv_to_html(cv: StandardCV) -> str:
    html = [
        "<!DOCTYPE html>",
        "<html lang='es'>",
        "<head>",
        '<meta charset="utf-8">',
        f"<title>CV - {esc(cv.candidate_name)}</title>",
        "<style>",
        "  body {",
        "    font-family: 'Georgia', 'Times New Roman', serif;",
        "    margin: 50px auto;",
        "    max-width: 750px;",
        "    color: #000;",
        "    line-height: 1.5;",
        "    font-size: 15px;",
        "  }",
        "  h1 {",
        "    text-align: center;",
        "    font-size: 1.8em;",
        "    letter-spacing: 1px;",
        "    margin-bottom: 4px;",
        "    font-weight: bold;",
        "  }",
        "  .contact {",
        "    text-align: center;",
        "    font-size: 0.9em;",
        "    margin-bottom: 20px;",
        "  }",
        "  .contact a { color: #000; text-decoration: none; border-bottom: 1px solid #000; }",
        "  hr.top-rule {",
        "    border: none;",
        "    border-top: 1px solid #000;",
        "    margin: 0 0 20px 0;",
        "  }",
        "  h2 {",
        "    font-size: 1em;",
        "    text-transform: uppercase;",
        "    letter-spacing: 1.5px;",
        "    border-bottom: 1px solid #000;",
        "    padding-bottom: 3px;",
        "    margin-top: 26px;",
        "    margin-bottom: 10px;",
        "  }",
        "  h3 {",
        "    display: flex;",
        "    justify-content: space-between;",
        "    font-size: 1em;",
        "    font-weight: bold;",
        "    margin-bottom: 2px;",
        "  }",
        "  h3 .role-company { font-style: italic; font-weight: normal; }",
        "  .entry { margin-bottom: 14px; }",
        "  .entry-header {",
        "    display: flex;",
        "    justify-content: space-between;",
        "    font-weight: bold;",
        "  }",
        "  .entry-sub { font-style: italic; margin-bottom: 4px; }",
        "  ul { margin-top: 4px; margin-bottom: 0; padding-left: 20px; }",
        "  li { margin-bottom: 3px; }",
        "  p { margin-top: 4px; }",
        "  .skills, .list-line { margin-top: 4px; }",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{esc(cv.candidate_name)}</h1>",
    ]

    # Contacto (centrado, estilo Harvard)
    contact_parts = []
    if cv.email:
        contact_parts.append(esc(cv.email))
    if cv.phone:
        contact_parts.append(esc(cv.phone))
    if cv.location:
        contact_parts.append(esc(cv.location))
    if cv.linkedin_url:
        contact_parts.append(f'<a href="{esc(_url(cv.linkedin_url))}">LinkedIn</a>')
    if cv.portfolio_url:
        contact_parts.append(f'<a href="{esc(_url(cv.portfolio_url))}">Portfolio</a>')

    if contact_parts:
        html.append(f'<div class="contact">{" | ".join(contact_parts)}</div>')

    html.append('<hr class="top-rule">')

    # Perfil Profesional
    if cv.professional_summary:
        html.append("<h2>Perfil Profesional</h2>")
        html.append(f"<p>{esc(cv.professional_summary)}</p>")

    # Experiencia Laboral
    if cv.work_history:
        html.append("<h2>Experiencia Laboral</h2>")
        for work in cv.work_history:
            duration_str = esc(work.duration)
            html.append('<div class="entry">')
            html.append(
                f'<div class="entry-header"><span>{esc(work.company)}</span>'
                f"<span>{duration_str}</span></div>"
            )
            html.append(f'<div class="entry-sub">{esc(work.role)}</div>')
            if work.responsibilities:
                html.append("<ul>")
                for resp in work.responsibilities:
                    html.append(f"  <li>{esc(resp)}</li>")
                html.append("</ul>")
            html.append("</div>")

    # Educación
    if cv.education:
        html.append("<h2>Educación</h2>")
        for edu in cv.education:
            html.append('<div class="entry">')
            html.append(
                f'<div class="entry-header"><span>{esc(edu.institution)}</span>'
                f"<span>{esc(edu.year)}</span></div>"
            )
            html.append(f'<div class="entry-sub">{esc(edu.degree)}</div>')
            html.append("</div>")

    # Habilidades Técnicas
    if cv.technical_skills:
        html.append("<h2>Habilidades Técnicas</h2>")
        html.append(f'<p class="skills">{esc(", ".join(cv.technical_skills))}</p>')

    # Certificaciones
    if cv.certifications:
        html.append("<h2>Certificaciones</h2>")
        for cert in cv.certifications:
            html.append('<div class="entry">')
            html.append(
                f'<div class="entry-header"><span>{esc(_join(cert.name, cert.issuing_organization, sep=" — "))}</span>'
                f"<span>{esc(cert.issue_date)}</span></div>"
            )
            html.append("</div>")

    # Idiomas
    if cv.languages:
        html.append("<h2>Idiomas</h2>")
        html.append(f'<p class="list-line">{esc(", ".join(cv.languages))}</p>')

    # Intereses
    if cv.interests:
        html.append("<h2>Intereses</h2>")
        html.append(f'<p class="list-line">{esc(", ".join(cv.interests))}</p>')

    html.extend(["</body>", "</html>"])
    return "\n".join(html)
