"""
CVertidor: herramienta de Inteligencia Artificial para adaptar currículums
(CV) a ofertas de trabajo específicas mediante flujos de RAG local y un grafo
cíclico verificable (LangGraph + evaluador gratuito de OpenRouter).

Módulos:
    config       → credenciales y parámetros (variables de entorno)
    models       → esquemas de dominio en Pydantic (StandardCV, ...)
    processor    → chunkeo y generación de Document del CV
    database     → Supabase/pgvector: ingest, MMR, persistencia de resultados
    prompts      → plantillas de prompt centralizadas
    llm          → clientes de IA (Gemini: parse + generador estructurado)
    openrouter   → evaluador gratuito (modelo :free vía requests)
    graph        → grafo cíclico LangGraph (AgentState, nodos, enrutador)
    main         → orquestador y punto de entrada por consola
"""

from cvertidor.graph import AgentState, build_graph, run_agent
from cvertidor.models import Certification, Education, StandardCV, WorkExperience

__all__ = [
    "AgentState",
    "build_graph",
    "run_agent",
    "StandardCV",
    "WorkExperience",
    "Education",
    "Certification",
]
