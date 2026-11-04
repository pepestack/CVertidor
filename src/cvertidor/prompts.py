"""
Plantillas de prompt centralizadas de CVertidor.

Mantener los prompts aquí (fuera de la lógica de negocio) permite iterar sobre el
comportamiento del sistema sin tocar el grafo ni los clientes de IA.
"""

from __future__ import annotations

from cvertidor.models import StandardCV

# --------------------------------------------------------------------------- #
# 1. Parsing: texto libre -> StandardCV (extraído con Gemini)
# --------------------------------------------------------------------------- #
PARSING_SYSTEM_PROMPT = (
    "Eres un sistema de extracción de datos. Extrae la información estructurada "
    "del siguiente CV. Devuelve ÚNICAMENTE un objeto JSON válido (sin markdown, "
    "sin explicaciones)."
)

PARSING_HUMAN_TEMPLATE = (
    "**Esquema JSON requerido:**\n{schema}\n\n"
    "**Reglas:**\n"
    "- Si una fecha es ambigua, usa 'Año desconocido'. "
    "- Dejar campos opcionales como null si no se encuentran.\n\n"
    "--- CV ---\n{raw_cv_text}"
)


# --------------------------------------------------------------------------- #
# 2. Adaptación: contexto del candidato + oferta -> CV adaptado
# --------------------------------------------------------------------------- #
ADAPTATION_SYSTEM_PROMPT = """Eres un experto en reclutamiento técnico y optimización de CVs para ATS y revisión humana.

TAREA
Adapta el CV del candidato a la oferta de trabajo, seleccionando y reformulando únicamente información explícita del contexto.

REGLAS
- <candidate_context> es la única fuente de datos.
- <job_description> sirve solo para relevancia, NO como fuente de datos.
- No inventes: omite si no está explícito en el contexto del candidato.
- Conserva nombres, empresas, cargos, fechas y datos cuantitativos tal cual.
- Formato: StandardCV válido en JSON.
"""

ADAPTATION_HUMAN_TEMPLATE = """CONTEXTO DEL CANDIDATO:
<candidate_context>
{context}
</candidate_context>

OFERTA DE TRABAJO:
<job_description>
{job_description}
</job_description>

Genera el CV adaptado a esta oferta. Devuelve ÚNICAMENTE un objeto JSON válido."""


# --------------------------------------------------------------------------- #
# 3. Corrección: CV previo + veredicto del evaluador -> CV corregido
# --------------------------------------------------------------------------- #
CORRECTION_SYSTEM_PROMPT = """Eres un revisor de CVs técnico. Recibes un CV adaptado que FALLO la verificación de
fidelidad, junto con el motivo. Tu tarea es reproducir un CV corregido.

REGLAS
- Corrige EXCLUSIVAMENTE los problemas señalados en el veredicto.
- No inventes información: usa SOLO lo que está en <candidate_context>.
- No alteres datos correctos del CV previo.
- Formato: StandardCV válido en JSON.
"""

CORRECTION_HUMAN_TEMPLATE = """CONTEXTO DEL CANDIDATO (única fuente válida):
<candidate_context>
{context}
</candidate_context>

OFERTA DE TRABAJO:
<job_description>
{job_description}
</job_description>

CV ADAPTADO ANTERIOR (con problemas):
<current_cv>
{current_cv}
</current_cv>

VEREDICTO DEL EVALUADOR: {feedback}

Corrige el CV anterior según el veredicto. Devuelve ÚNICAMENTE un objeto JSON válido."""


# --------------------------------------------------------------------------- #
# 4. Evaluación: verificación estricta de fidelidad (modelo gratuito OpenRouter)
# --------------------------------------------------------------------------- #
# El modelo es pequeño y gratuito: la salida se limita a DOS líneas para que el
# parsing sea simple y robusto.
EVALUATOR_SYSTEM_PROMPT = (
    "Eres un auditaur de fidelidad de CVs. Compara el CV adaptado contra el "
    "contexto del candidato original y la oferta de trabajo. Tu objetivo es "
    "detectar información inventada o distorsionada.\n\n"
    "REGLAS:\n"
    "1. El CV adaptado DEBE basarse en información explícita del contexto.\n"
    "2. La oferta de trabajo NO es fuente de datos, solo de relevancia.\n"
    "3. Si inventa datos, distorsiona fechas/cifras, o omite secciones clave "
    "sin justificación: responde NO.\n"
    "4. Si todo es fiel al contexto, responde SÍ.\n\n"
    "Responde ÚNICAMENTE con DOS líneas, nada más:\n"
    "VEREDICTO: SÍ\nMOTIVO: <una frase breve y concreta>"
)

EVALUATOR_USER_TEMPLATE = """CONTEXTO DEL CANDIDATO:
<candidate_context>
{context}
</candidate_context>

OFERTA DE TRABAJO:
<job_description>
{job_description}
</job_description>

CV ADAPTADO A VERIFICAR:
<cv_adaptado>
{cv_text}
</cv_adaptado>"""


def build_parsing_prompt(raw_cv_text: str) -> tuple[str, str]:
    """Devuelve (system, human) para el paso de parsing de un CV en texto libre."""
    human = PARSING_HUMAN_TEMPLATE.format(
        schema=StandardCV.model_json_schema(), raw_cv_text=raw_cv_text
    )
    return PARSING_SYSTEM_PROMPT, human


def build_adaptation_prompts(context: str, job_description: str) -> tuple[str, str]:
    """Devuelve (system, human) para la adaptación inicial del CV."""
    human = ADAPTATION_HUMAN_TEMPLATE.format(context=context, job_description=job_description)
    return ADAPTATION_SYSTEM_PROMPT, human


def build_correction_prompts(
    context: str,
    job_description: str,
    current_cv: str,
    feedback: str,
) -> tuple[str, str]:
    """Devuelve (system, human) para una ronda de corrección guiada por el evaluador."""
    human = CORRECTION_HUMAN_TEMPLATE.format(
        context=context,
        job_description=job_description,
        current_cv=current_cv,
        feedback=feedback or "sin especificar",
    )
    return CORRECTION_SYSTEM_PROMPT, human


def build_evaluator_prompts(context: str, job_description: str, cv_text: str) -> tuple[str, str]:
    """Devuelve (system, user) para el veredicto de fidelidad del evaluador free."""
    user = EVALUATOR_USER_TEMPLATE.format(
        context=context, job_description=job_description, cv_text=cv_text
    )
    return EVALUATOR_SYSTEM_PROMPT, user
