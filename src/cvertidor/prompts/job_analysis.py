"""Prompt de extracción de requisitos de la oferta a JobRequirements."""

from langchain_core.prompts import ChatPromptTemplate

JOB_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Eres un analista de ofertas de trabajo. Conviertes el texto de una oferta en el esquema JobRequirements.

REGLAS
- Usa únicamente lo que dice <job_description>; no agregues requisitos típicos del cargo que no aparecen.
- `must_have`: requisitos obligatorios o excluyentes. `nice_to_have`: los marcados como deseables, valorables o un plus.
- Escribe cada requisito en 1 a 4 palabras, con el nombre de la herramienta o habilidad (ej. "Python", "Google BigQuery", "modelamiento dimensional"), sin años de experiencia.
- Escribe los requisitos en el mismo idioma de la oferta, sin traducirlos.
- No repitas un requisito en ambas listas.
- `language`: el idioma en que está redactada la oferta ('es' o 'en'), no los idiomas que pide."""
    ),
    ("human", "<job_description>\n{job_description}\n</job_description>"),
])
