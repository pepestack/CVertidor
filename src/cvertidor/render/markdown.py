"""Generador de Markdown para un StandardCV."""

from cvertidor.render.utils import _join, _url
from cvertidor.schemas.cv import StandardCV


def cv_to_markdown(cv: StandardCV) -> str:
    md = []

    # Encabezado / Datos de contacto
    md.append(f"# {cv.candidate_name}\n")
    contact_info = []
    if cv.email: contact_info.append(f"**Email:** {cv.email}")
    if cv.phone: contact_info.append(f"**Teléfono:** {cv.phone}")
    if cv.location: contact_info.append(f"**Ubicación:** {cv.location}")
    if cv.linkedin_url: contact_info.append(f"**LinkedIn:** [{cv.linkedin_url}]({_url(cv.linkedin_url)})")
    if cv.portfolio_url: contact_info.append(f"**Portfolio:** [{cv.portfolio_url}]({_url(cv.portfolio_url)})")

    if contact_info:
        md.append(" | ".join(contact_info) + "\n\n---\n")


    # Perfil
    if cv.professional_summary:
        md.append("## Perfil Profesional\n")
        md.append(f"{cv.professional_summary}\n")

    # Experiencia Laboral
    if cv.work_history:
        md.append("## Experiencia Laboral\n")
        for work in cv.work_history:
            duration_str = f" ({work.duration})" if work.duration else ""
            md.append(f"### {work.role} - {work.company}{duration_str}")
            for resp in work.responsibilities:
                md.append(f"- {resp}")
            md.append("")

    # Educación
    if cv.education:
        md.append("## Educación\n")
        for edu in cv.education:
            md.append(f"- {_join(edu.degree, edu.institution, edu.year)}")
        md.append("")

    # Habilidades Técnicas
    if cv.technical_skills:
        md.append("## Habilidades Técnicas\n")
        md.append(", ".join(cv.technical_skills) + "\n")

    # Certificaciones
    if cv.certifications:
        md.append("## Certificaciones\n")
        for cert in cv.certifications:
            md.append(
                f"- {_join(cert.name, cert.issuing_organization, cert.issue_date)}"
            )
        md.append("")

    # Idiomas
    if cv.languages:
        md.append("## Idiomas\n")
        for lang in cv.languages:
            md.append(f"- {lang}")
        md.append("")

    # Intereses
    if cv.interests:
        md.append("## Intereses\n")
        for interest in cv.interests:
            md.append(f"- {interest}")
        md.append("")

    return "\n".join(md)
