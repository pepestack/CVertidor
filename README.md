# CVertidor

Herramienta de **Inteligencia Artificial** para **adaptar currículums (CV) a ofertas
de trabajo** usando RAG local (Gemini + pgvector en Supabase) y un **grafo cíclico
verificable** con LangGraph. El grafo incorpora un **evaluador lógico 100% gratuito**
(modelo `meta-llama/llama-3-8b-instruct:free` de OpenRouter) que decide si el CV
adaptado es **fidel al contexto** y, en caso negativo, devuelve el flujo al nodo de
corrección hasta lograr la fidelidad o un `max_iterations`.

> Proyecto universitario — DUOC UC · OPT-ISY0101 · Ingeniería de Soluciones con IA.

---

## Arquitectura

```
src/cvertidor/
├── config.py       # Credenciales y parámetros (variables de entorno / .env)
├── models.py       # Esquemas Pydantic: StandardCV, WorkExperience, Education, Certification
├── processor.py    # Chunkeo y creación de Document (create_cv_documents)
├── database.py     # Supabase/pgvector: ingest, MMR, persistencia de resultados
├── prompts.py      # Plantillas de prompt centralizadas
├── llm.py          # Gemini: parse a StandardCV + generador estructurado
├── openrouter.py   # Evaluador gratuito vía API REST nativa (requests)
├── graph.py        # Grafo cíclico LangGraph (AgentState, nodos, enrutador)
├── main.py         # Orquestador / punto de entrada por consola
└── __main__.py     # Soporte: python -m cvertidor
```

### Flujo del grafo

```
START → generate (Gemini) → evaluate (OpenRouter :free)
                                   │
                  SÍ ────────────┬── NO (y quedan ciclos)
                  ▼              ▼
                save (Supabase)  correct (Gemini) ──► evaluate
                  ▼
                END
```

---

## Requisitos

- Python **3.12+**
- [uv](https://docs.astral.sh/uv/) para gestionar dependencias y entorno virtual
- API keys: **Google AI Studio** (Gemini) y **OpenRouter** (evaluador free)
- Proyecto **Supabase** habilitado con la extensión `vector` (pgvector)

---

## Inicialización con `uv`

```bash
# 1) Inicializar el proyecto (ya hecho en esta carpeta; si partes de cero:
#    uv init --python 3.12 cvertidor --no-readme

# 2) Crear el entorno virtual con Python 3.12
uv venv --python 3.12

# 3) Instalar dependencias (ya declaradas en pyproject.toml)
uv sync

# 4) (Opcional) dependencias de desarrollo: linter, formatter y notebooks
uv sync --extra dev
```

### Agregar / actualizar dependencias manualmente

```bash
uv add langchain-core langchain-classic langchain-community
uv add langchain-text-splitters langchain-google-genai
uv add "langchain-postgres[psycopg]"
uv add "psycopg[binary]" pgvector sqlalchemy
uv add pydantic google-genai langgraph
uv add python-dotenv requests

# Dev tooling
uv add --dev pytest ruff mypy
```

---

## Configuración

```bash
cp .env.example .env
# Edita .env y rellena: GOOGLE_API_KEY, SUPABASE_URL, OPENROUTER_API_KEY
```

---

## Uso

```bash
# Modo demo (CV y oferta incluidos)
uv run cvertidor --demo

# Con tu propio CV y oferta
uv run cvertidor --cv mi_cv.txt --job oferta.txt

# Desde texto en línea
uv run cvertidor --cv-text "Juan Pérez ..." --job-text "Buscamos un Data Engineer ..."

# Equivalente por módulo
uv run python -m cvertidor --demo
```

---

## Notas

- Los nombres de modelo `GEMINI_*` y `OPENROUTER_EVAL_MODEL` son **configurables**
  desde `.env`. Ajusta `GEMINI_GEN_MODEL` / `GEMINI_PARSE_MODEL` a un modelo
  Gemini disponible en tu plan (por ejemplo `gemini-1.5-flash`).
- El evaluador free de OpenRouter tiene un **contexto pequeño** y **no soporta
  structured output**; por eso el prompt es estricto y el parse es tolerante.
- En caso de falla de red en el evaluador, el grafo **forza una corrección** y
  un `max_iterations` **impide bucles infinitos**.
