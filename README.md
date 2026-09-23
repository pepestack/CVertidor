# CVertidor

[![Abrir en Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pepestack/CVertidor/blob/master/CVertidor.ipynb)

CVertidor es una herramienta de inteligencia artificial generativa que adapta un currículum a una oferta de trabajo mediante un flujo de **RAG (Retrieval-Augmented Generation)**. Extrae la información del candidato, la divide en fragmentos, genera embeddings y la almacena temporalmente en PostgreSQL con `pgvector`. Después recupera los fragmentos más relevantes para generar un CV orientado a la oferta, sin inventar experiencias ni datos.

El resultado se puede visualizar y guardar en dos formatos:

- Markdown (`cv_<candidato>.md`)
- HTML (`cv_<candidato>.html`)

El proyecto está implementado principalmente en el notebook [`CVertidor.ipynb`](./CVertidor.ipynb), por lo que puede ejecutarse en Google Colab o localmente con Jupyter.

## ¿Por qué usar CVertidor?

- **Adaptación enfocada:** prioriza experiencia, habilidades y certificaciones relacionadas con la oferta.
- **Trazabilidad del contenido:** el prompt restringe la generación a la información explícita del CV del candidato.
- **Búsqueda semántica:** usa embeddings y pgvector para recuperar los fragmentos más útiles.
- **Salida reutilizable:** genera un documento Markdown editable y una versión HTML lista para visualizar.
- **Fallback de embeddings:** intenta usar embeddings de Google y puede cambiar a `sentence-transformers/all-MiniLM-L6-v2` si el proveedor principal no está disponible.
- **Validación estructurada:** usa modelos Pydantic para representar experiencia, educación, habilidades y certificaciones.

## Arquitectura del flujo

1. Se ingresan el currículum y la oferta de trabajo como texto.
2. Gemini estructura el currículum en un esquema `StandardCV`.
3. El contenido se transforma en documentos y fragmentos semánticos.
4. Los embeddings se guardan en una colección de PostgreSQL/pgvector.
5. Un recuperador MMR selecciona información relevante para la oferta.
6. Gemini genera el CV adaptado y lo valida contra el esquema.
7. El resultado se muestra y se guarda como Markdown y HTML.

> La aplicación y el flujo son locales, pero requieren un proveedor de modelo generativo/embeddings y una base de datos PostgreSQL accesible. No se debe ingresar información sensible real sin evaluar previamente las políticas de privacidad del proveedor utilizado.

## Requisitos previos

- Python 3.10 o superior.
- Una clave de API de Google con acceso a los modelos Gemini.
- PostgreSQL con la extensión `pgvector`.
- Docker Desktop, si se quiere levantar PostgreSQL localmente con el archivo [`compose.yml`](./compose.yml).
- Jupyter Notebook o Google Colab.

## Instalación y configuración local

### 1. Clonar el repositorio

```bash
git clone https://github.com/pepestack/CVertidor.git
cd CVertidor
```

### 2. Crear un entorno virtual e instalar dependencias

En Windows:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

En macOS o Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Crear el archivo de variables de entorno

Copia el archivo de ejemplo:

```powershell
Copy-Item .env.example .env
```

Edita `.env` y reemplaza los valores de ejemplo:

```dotenv
POSTGRES_DB=cvertidor_db
POSTGRES_USER=admin
POSTGRES_PASSWORD=una_clave_segura
DB_URL=postgresql://admin:una_clave_segura@localhost:5432/cvertidor_db
API_KEY=tu_clave_de_google
```

`DB_URL` debe ser una URI PostgreSQL válida. En un entorno administrado, como Supabase, usa la URI de conexión proporcionada por el servicio y asegúrate de que `pgvector` esté habilitado.

### 4. Levantar PostgreSQL con pgvector

El repositorio incluye una configuración de Docker Compose que crea la base de datos y activa la extensión `vector` mediante [`init.sql`](./init.sql). Ejecuta este comando después de crear `.env`:

```bash
docker compose up -d
```

La base de datos queda disponible en `localhost:5432` con los valores definidos en [`.env.example`](./.env.example). Para detenerla:

```bash
docker compose down
```

Los datos persisten en el volumen de Docker `pgdata`. Para eliminar también ese volumen, usa `docker compose down -v`.

## Uso

### Ejecutar localmente

Inicia Jupyter desde el entorno virtual:

```bash
pip install jupyter
jupyter notebook
```

Abre [`CVertidor.ipynb`](./CVertidor.ipynb) y ejecuta las celdas en orden. En la celda de entrada, reemplaza `tu_curriculum` y `oferta_de_trabajo` por tus textos:

```python
tu_curriculum = """
Nombre: Ana Pérez
Experiencia: ...
Habilidades: Python, SQL, PostgreSQL
"""

oferta_de_trabajo = """
Buscamos una persona desarrolladora Python con experiencia en APIs y PostgreSQL.
"""
```

El notebook valida que ambos textos tengan una longitud mínima. Si el contenido es demasiado corto, utiliza datos de ejemplo incluidos en el notebook.

Al finalizar, se muestran las versiones Markdown y HTML, y se crean dos archivos en el directorio de trabajo:

```text
cv_ana_pérez.md
cv_ana_pérez.html
```

### Ejecutar en Google Colab

También puedes abrir directamente el notebook con el botón de la parte superior. En Colab:

1. Instala las dependencias ejecutando la celda de instalación.
2. Configura `DB_URL` y `GOOGLE_API_KEY` como secretos de Colab.
3. Asegúrate de que la base PostgreSQL sea accesible desde Internet y tenga `pgvector` habilitado.
4. Ejecuta las celdas en orden.
5. Descarga los archivos HTML o Markdown generados desde el entorno de Colab.

Para una ejecución local, el notebook carga `DB_URL` y `API_KEY` desde `.env`; en Colab utiliza `DB_URL` y `GOOGLE_API_KEY` desde `userdata`.

## Estructura del repositorio

```text
.
├── CVertidor.ipynb  # Flujo completo de extracción, RAG y generación del CV
├── compose.yml      # PostgreSQL con pgvector para desarrollo local
├── init.sql         # Activación de la extensión vector
├── requirements.txt # Dependencias de Python
├── .env.example     # Plantilla de configuración local
└── .gitignore
```

## Consideraciones y limitaciones

- El proyecto no incluye una interfaz web ni un servicio HTTP; la interfaz actual es el notebook.
- La generación depende de una API de Google cuando se usa Gemini como modelo principal.
- El fallback de embeddings requiere descargar el modelo de Hugging Face la primera vez.
- El flujo elimina la colección vectorial anterior antes de insertar el CV actual. Está pensado para trabajar con un CV por ejecución, no como repositorio multiusuario.
- No subas `.env`, claves API ni currículums reales al repositorio. `.env` ya está excluido por [`.gitignore`](./.gitignore).

## Contribuir

Las mejoras son bienvenidas. Para contribuir:

1. Crea una rama descriptiva a partir de `master`.
2. Realiza cambios pequeños y enfocados.
3. Verifica el notebook y la conexión a PostgreSQL/pgvector antes de abrir un pull request.
4. Describe en el pull request qué cambió y cómo se validó.

## Licencia

Este repositorio no incluye actualmente un archivo `LICENSE`. Consulta con los mantenedores antes de redistribuir el código o incorporarlo en otro proyecto.
