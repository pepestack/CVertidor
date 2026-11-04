#!/usr/bin/env python
# coding: utf-8

# # **CVertidor**
# ## Herramienta impulsada por Inteligencia Artificial para adaptar curriculums mediante flujos de RAG local
# ### Ingeniería de Soluciones con Inteligencia Artificial - OPT-ISY0101 | DUOC UC
# 
# ---
# 
# | Campo | Detalle |
# |---|---|
# | **Integrante** | _José Valentín Calderón González_ |
# | **Integrante** | _Mauricio Antonio Hidalgo Rivera_ |
# | **Sección** | _010D_ |
# | **Docente** | _Jaime Andres Menendez Silva_ |
# | **Fecha de entrega** | _RELLENAR_ |
# 
# ---

# ## 📦 Instalación de dependencias
# Descomenta y ejecuta la primera celda para instalar las librerías necesarias.

# In[ ]:


get_ipython().system('pip install -q langchain-text-splitters langchain-core langchain-google-genai langchain-classic langchain-community pydantic langchain-postgres psycopg[binary] google-genai')


# ## 🧠 Importación de librerías

# In[ ]:


import os
import json
import re
import time
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from google import genai
from pydantic import BaseModel, Field

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, trim_messages
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.outputs import LLMResult
from langchain_core.runnables import RunnablePassthrough

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

import psycopg

print('✅ Librerías importadas.')


# ## 🔑 Configuración de credenciales (Colab Secrets)

# In[ ]:


from google.colab import userdata

api_key_google = userdata.get('GOOGLE_API_KEY')
os.environ['GOOGLE_API_KEY'] = api_key_google

supa_url = userdata.get('SUPABASE_URL')
os.environ['SUPABASE_URL'] = supa_url

import os
print('✅ GOOGLE_API_KEY configurada.')
print('✅ SUPABASE_URL configurada.')


# ## 🌐 Cliente Google AI Studio

# In[ ]:


google_ai_client = genai.Client(api_key=api_key_google)
print('✅ Cliente Google AI Studio configurado.')


# ## 🗄️ Conexión a Supabase e inicialización de pgvector

# In[ ]:


conn = psycopg.connect(supa_url)
print('✅ Conectado a Supabase!')

with conn.cursor() as cur:
    cur.execute("SELECT version();")
    print('Versión Postgres:', cur.fetchone())

with conn.cursor() as cur:
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
conn.commit()
print('¡Extensión vector activada!')

with conn.cursor() as cur:
    cur.execute("SELECT '[1,2,3]'::vector;")
    print('Prueba vector:', cur.fetchone())

conn.close()


# ## 📐 Modelos de datos (Pydantic)

# In[ ]:


class Certification(BaseModel):
    name: str = Field(description='Nombre de la certificación, licencia o acreditación.')
    issuing_organization: Optional[str] = Field(None, description='Organización que emitió la certificación.')
    issue_date: Optional[str] = Field(None, description='Fecha de emisión u vigencia.')

class WorkExperience(BaseModel):
    company: str = Field(description='Empresa u organización donde trabajó.')
    role: str = Field(description='Cargo o puesto desempeñado.')
    duration: Optional[str] = Field(None, description='Período de trabajo, inicio y término.')
    responsibilities: List[str] = Field(default_factory=list, description='Responsabilidades y logros del cargo.')

class Education(BaseModel):
    institution: str = Field(description='Universidad u otra Institución educativa.')
    degree: str = Field(description='Título, grado, carrera o programa académico.')
    year: Optional[str] = Field(None, description='Año o período de graduación.')

class StandardCV(BaseModel):
    candidate_name: str = Field(description='Nombre completo del candidato')
    email: Optional[str] = Field(default=None, description='Email')
    phone: Optional[str] = Field(default=None, description='Teléfono de contacto')
    location: Optional[str] = Field(default=None, description='Ciudad, región o país de ubicación.')
    linkedin_url: Optional[str] = Field(default=None, description='URL de perfil de LinkedIn')
    portfolio_url: Optional[str] = Field(default=None, description='URL de GitHub o portafolio')

    education: List[Education] = Field(default_factory=list, description='Formación académica.')
    work_history: List[WorkExperience] = Field(default_factory=list, description='Experiencia laboral.')

    technical_skills: Optional[List[str]] = Field(default=None, description='Habilidades técnicas.')
    certifications: Optional[List[Certification]] = Field(default=None, description='Certificaciones o licencias')
    languages: Optional[List[str]] = Field(default=None, description='Idiomas y nivel de dominio.')
    interests: Optional[List[str]] = Field(default=None, description='Intereses y actividades extracurriculares.')


# ## 📄 Procesador y chunkeo de documentos

# In[ ]:


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=['\n- ', '\n* ', '\n', '. ', ' ']
)

def create_cv_documents(parsed_cv: StandardCV) -> List[Document]:
    candidate_id = parsed_cv.candidate_name.lower().replace(' ', '_')
    candidate_name = parsed_cv.candidate_name
    documents = []

    for exp in parsed_cv.work_history:
        header = (
            f'Candidato: {candidate_name}\n'
            f'Rol: {exp.role} - {exp.company}\n'
            f'Duración: {exp.duration or "N/A"}\n'
            'Responsabilidades:\n'
        )
        resp = '\n- '.join(exp.responsibilities)
        resp_content = f'- {resp}'
        base_meta = {'candidate_id': candidate_id, 'section': 'work_history', 'company': exp.company, 'role': exp.role}

        if len(resp_content) <= 500:
            documents.append(Document(page_content=header + resp_content, metadata={**base_meta, 'sub_chunk_index': 0}))
        else:
            temp = Document(page_content=resp_content, metadata=base_meta)
            for i, sub in enumerate(text_splitter.split_documents([temp])):
                sub.page_content = header + sub.page_content
                sub.metadata['sub_chunk_index'] = i
                documents.append(sub)

    for edu in parsed_cv.education:
        content = f'Candidato: {candidate_name}\nEducacion: {edu.degree} - {edu.institution}\nAño: {edu.year or "N/A"}'
        documents.append(Document(page_content=content, metadata={'candidate_id': candidate_id, 'section': 'education', 'institution': edu.institution}))

    if parsed_cv.technical_skills:
        documents.append(Document(page_content=f'Candidato: {candidate_name}\nHabilidades técnicas: {", ".join(parsed_cv.technical_skills)}', metadata={'candidate_id': candidate_id, 'section': 'technical_skills'}))

    if parsed_cv.certifications:
        certs = '; '.join(f"{c.name} ({c.issuing_organization or 'N/A'}) {c.issue_date or ''}".strip() for c in parsed_cv.certifications)
        documents.append(Document(page_content=f'Candidato: {candidate_name}\nCertificaciones: {certs}', metadata={'candidate_id': candidate_id, 'section': 'certifications'}))

    if parsed_cv.languages:
        langs = ', '.join(parsed_cv.languages)
        documents.append(Document(page_content=f'Candidato: {candidate_name}\nIdiomas: {langs}', metadata={'candidate_id': candidate_id, 'section': 'languages'}))

    if parsed_cv.interests:
        interests_list = '\n- '.join(parsed_cv.interests)
        documents.append(Document(page_content=f'Candidato: {candidate_name}\nIntereses:\n- {interests_list}', metadata={'candidate_id': candidate_id, 'section': 'interests'}))

    return documents


# ## 🔍 Parsing de CV con Google AI Studio

# In[ ]:


raw_cv_text = '''
Juan Pérez
Email: juan.perez@email.com | Tel: +56912345678 | Santiago, Chile
LinkedIn: linkedin.com/in/juanperez

Experiencia Laboral:
- Senior Data Engineer en TechCorp (2021 - Presente):
  Lideré la migración de pipelines ETL a Databricks, reduciendo tiempos de procesamiento en un 40%.
  Implementé arquitecturas de búsqueda vectorial y RAG utilizando PostgreSQL (pgvector) y LangChain.

Educación:
- Ingeniería Civil Informática, Universidad de Chile (2015 - 2020)

Habilidades Técnicas: Python, SQL, PostgreSQL, LangChain, Spark, Docker, GCP
Idiomas: Español (Nativo), Inglés (Avanzado)
Certificaciones: AWS Certified Solutions Architect (AWS) 2023
'''

response = google_ai_client.models.generate_content(
    model='gemini-3.6-flash',
    contents=(
        'Eres un sistema de extracción de datos. Extrae la información estructurada del siguiente CV.'
        " Devuelve ÚNICAMENTE un objeto JSON válido (sin markdown, sin explicaciones).\n\n"
        f'**Esquema JSON requerido:**\n{StandardCV.model_json_schema()}\n\n'
        '**Reglas:**\n'
        "- Si una fecha es ambigua, usa 'Año desconocido'. "
        '- Dejar campos opcionales como null si no se encuentran.\n\n'
        '--- CV ---\n'
        f'{raw_cv_text}'
    ),
    config={
        'response_mime_type': 'application/json',
        'response_schema': StandardCV,
        'temperature': 0.0,
    }
)

text = response.text.strip()
if text.startswith('```'):
    text = re.sub(r'^```(?:json)?\n?|```\s*$', '', text, flags=re.MULTILINE).strip()

parsed_cv: StandardCV = StandardCV.model_validate(json.loads(text))

print(f'✅ Extracción completada para: {parsed_cv.candidate_name}')
print(f'Habilidades:  {parsed_cv.technical_skills}')
print(f'Certificaciones: {parsed_cv.certifications}')


# In[ ]:


candidate_id = parsed_cv.candidate_name.lower().replace(' ', '_')

# Embedding
embeddings = GoogleGenerativeAIEmbeddings(
    model='models/gemini-embedding-2',
    output_dimensionality=768,
    google_api_key=api_key_google
)

documents = create_cv_documents(parsed_cv)
print(f'📄 Chunks generados: {len(documents)} para {candidate_id}')

# Guardar en Supabase PGVector
connection_string = supa_url.replace('postgresql://', 'postgresql+psycopg://', 1) if supa_url.startswith('postgresql://') else supa_url
COLLECTION_NAME = 'cv_embeddings'

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=connection_string,
    use_jsonb=True
)
vector_store.delete_collection()

vector_store = PGVector.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    connection=connection_string,
    use_jsonb=True
)

print(f'✅ Vectores insertados para {candidate_id}')


# ## 💬 RAG Pipeline (Gemini)

# In[ ]:


def format_docs(docs):
    return '\n\n'.join(f"--- Chunk ({doc.metadata.get('section', 'N/A')}) ---\n{doc.page_content}" for doc in docs)

retriever = vector_store.as_retriever(
    search_type='mmr',
    search_kwargs={
        'k': 6,
        'fetch_k': 20,
        'lambda_mult': 0.7,
        'filter': {'candidate_id': candidate_id}
    }
)

llm = ChatGoogleGenerativeAI(model='gemini-3.6-flash', temperature=0.1, google_api_key=api_key_google)
structured_llm = llm.with_structured_output(StandardCV)

prompt = ChatPromptTemplate.from_messages([
    ('system', '''Eres un experto en reclutamiento técnico y optimización de CVs para ATS y revisión humana.

TAREA
Adapta el CV del candidato a la oferta de trabajo, seleccionando y reformulando únicamente información explícita del contexto.

REGLAS
- <candidate_context> es la única fuente de datos.
- <job_description> sirve solo para relevancia, NO como fuente de datos.
- No inventes: omite si no está explícito en el contexto del candidato.
- Conserva nombres, empresas, cargos, fechas y datos cuantitativos tal cual.
- Formato: StandardCV válido en JSON.
'''),
    ('human', '''CONTEXTO DEL CANDIDATO:
<candidate_context>
{context}
</candidate_context>

OFERTA DE TRABAJO:
<job_description>
{job_description}
</job_description>

Genera el CV adaptado a esta oferta. Devuelve ÚNICAMENTE un objeto JSON válido.'''  )
])

rag_chain = (
    {'context': retriever | format_docs, 'job_description': RunnablePassthrough()}
    | prompt
    | structured_llm
)


# ## ▶️ Ejecución de prueba

# In[ ]:


job_description_test = 'Buscamos un Data Engineer con experiencia en Python, construcción de pipelines de datos y bases de datos vectoriales como pgvector.'

print('⏳ Recuperando chunks relevantes...')
retrieved_docs = retriever.invoke(job_description_test)
for i, doc in enumerate(retrieved_docs, 1):
    print(f'--- Chunk {i} [{doc.metadata["section"]}] ---')
    preview = doc.page_content[:150] + '...' if len(doc.page_content) > 150 else doc.page_content
    print(preview)

start_time = time.perf_counter()
response = rag_chain.invoke(job_description_test)
end_time = time.perf_counter()

print(f'\n✅ Respuesta generada en {end_time - start_time:.2f} segundos')
print('\n=== CV ADAPTADO ===')
print(response.model_dump_json(indent=2))

