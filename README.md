# HybridRank-RAG

HybridRank-RAG es un proyecto de recuperacion hibrida sobre documentos legales cubanos. El repo mantiene dos flujos paralelos:

- **Elementos originales**: usa los chunks generados por el loader/cleaner y los indexa en BM25 + Chroma.
- **Normas estructuradas**: parte de `normas.db`, crea fragmentos normativos estables, genera embeddings, construye indices propios y permite crear un dataset de evaluacion sobre esos fragmentos.

La separacion es intencional: los `chunk_id` del flujo original son aleatorios por corrida, mientras que los fragmentos normativos usan IDs deterministas.

## Rutas Principales

| Ruta | Uso |
| --- | --- |
| `.data/cleaned_content/cleaned_elements.json` | Elementos limpios del pipeline original. |
| `.data/norma_output/normas.db` | SQLite con normas estructuradas por `norma_processor.py`. |
| `.data/norma_output/norma_fragments.json` | Fragmentos normativos indexables. |
| `.data/embeddings/e5_elements_with_embeddings.json` | Embeddings del corpus original. |
| `.data/embeddings/e5_norma_fragments_with_embeddings.json` | Embeddings del corpus normativo. |
| `.data/bm25_index` | Indice BM25 del corpus original. |
| `.data/chroma` | Chroma del corpus original, coleccion `hybridrank_elements`. |
| `.data/bm25_norma_index` | Indice BM25 de fragmentos normativos. |
| `.data/chroma_normas` | Chroma normativo, coleccion `hybridrank_normas`. |
| `.data/evaluation/qrels.json` | Dataset de evaluacion del corpus original. |
| `.data/evaluation/norma_qrels.json` | Dataset de evaluacion del corpus normativo. |

## Componentes

### `src/data_preparation/text_cleaner.py`

Limpia los elementos extraidos por el loader. Produce documentos con `content`, `cleaned_content` y `metadata`, que luego pueden pasar a embeddings.

Uso:

```powershell
.\.venv\Scripts\python.exe -m src.data_preparation.text_cleaner
```

Argumentos:

| Argumento | Descripcion |
| --- | --- |
| `--input-file` | JSON con elementos extraidos. Default: `.data/processed_loader_output/all_raw_extracted_elements.json`. |
| `--output-file` | JSON de salida con `cleaned_content`. Default: `.data/cleaned_content/cleaned_elements.json`. |
| `--lemmatize` | Aplica lematizacion y filtrado de stopwords. |
| `--min-cleaned-length` | Longitud minima para conservar un elemento. Default: `10`. |
| `--batch-size` | Tamano de lote para limpieza. Default: `500`. |
| `--workers` | Numero de workers paralelos. Default: `cpu_count`. |

### `src/data_preparation/norma_processor.py`

Toma `cleaned_elements.json`, agrupa por gaceta, reconstruye texto, segmenta por codigos GOC y guarda normas estructuradas.

Salidas principales:

- `.data/norma_output/normas_by_gaceta.json`
- `.data/norma_output/normas.db`
- `.data/norma_output/processing_report.json`

Uso:

```powershell
.\.venv\Scripts\python.exe -m src.data_preparation.norma_processor
```

Argumentos:

| Argumento | Descripcion |
| --- | --- |
| `--input-file` | JSON con elementos limpios. |
| `--json-output` | JSON jerarquico de salida. |
| `--db-output` | SQLite de salida. |
| `--report-output` | Reporte de procesamiento. |
| `--skip-sqlite` | No genera SQLite. |
| `--skip-json` | No genera JSON jerarquico. |
| `--log-level` | `DEBUG`, `INFO`, `WARNING` o `ERROR`. |

### `src/data_preparation/norma_models.py`

Define los modelos de datos usados en la preparacion normativa:

- `NormaIdentity`: identidad canonica de una norma.
- `Norma`: norma extraida desde una gaceta.
- `Gaceta`: agrupacion de normas por edicion de gaceta.
- `NormaIndexFragment`: documento compatible con embeddings, BM25 y Chroma.
- Modelos de linking: `NormaReference`, `NormaRelationship`, `LinkingResult`.

`NormaIndexFragment` usa esta forma:

```json
{
  "content": "...",
  "cleaned_content": "...",
  "metadata": {
    "chunk_id": "fragment_id",
    "corpus_type": "normas",
    "norma_id": "...",
    "fragment_id": "..."
  },
  "embedding": []
}
```

### `src/data_preparation/norma_index_document_builder.py`

Convierte `normas.db` en fragmentos normativos indexables. Este es el puente entre las normas estructuradas y los indices BM25/Chroma.

Reglas principales:

- Excluye duplicados marcados como `superseded` por defecto.
- Usa IDs deterministas:
  `norma_id__gaceta_checksum_12__goc_code_or_ordinal__f000`.
- Si una norma tiene hasta `450` palabras, se indexa completa.
- Si es grande, se divide por estructura juridica: `CAPITULO`, `SECCION`, `ARTICULO`, `DISPOSICIONES`, `ANEXO`, `PRIMERO`, `SEGUNDO`, etc.
- Si un bloque sigue siendo grande, se divide en ventanas de `380` palabras con `60` de solapamiento.
- Cada fragmento conserva contexto:
  `Tipo Numero de Year - Organismo. Fragment label. Texto fragmento`.

Uso:

```powershell
.\.venv\Scripts\python.exe -m src.data_preparation.norma_index_document_builder
```

Argumentos:

| Argumento | Descripcion |
| --- | --- |
| `--db-path` | Ruta a `normas.db`. Default: `.data/norma_output/normas.db`. |
| `--output-file` | JSON de fragmentos. Default: `.data/norma_output/norma_fragments.json`. |
| `--include-superseded` | Incluye ocurrencias duplicadas/superseded. |
| `--log-level` | `DEBUG`, `INFO`, `WARNING` o `ERROR`. |

### `src/embedding_model/embedding_generator_e5.py`

Genera embeddings con `intfloat/multilingual-e5-base`. Usa prefijo E5 de pasaje y puede procesar elementos originales o fragmentos normativos, siempre que tengan `cleaned_content` y `metadata.chunk_id`.

Corpus original:

```powershell
.\.venv\Scripts\python.exe src\embedding_model\embedding_generator_e5.py --input-file .data\cleaned_content\cleaned_elements.json --output-file .data\embeddings\e5_elements_with_embeddings.json
```

Corpus normativo:

```powershell
.\.venv\Scripts\python.exe src\embedding_model\embedding_generator_e5.py --input-file .data\norma_output\norma_fragments.json --output-file .data\embeddings\e5_norma_fragments_with_embeddings.json --stats-file .data\embeddings\e5_norma_embedding_stats.json
```

Argumentos:

| Argumento | Descripcion |
| --- | --- |
| `--input-file` | JSON de entrada. Default: `.data/cleaned_content/cleaned_elements.json`. |
| `--output-file` | JSON de salida con embeddings. |
| `--stats-file` | JSON de estadisticas. |
| `--batch-size` | Tamano de lote. Default: `64`. |
| `--num-workers` | Procesos paralelos. Default: `1`. |
| `--fp16` | Usa float16 si hay GPU compatible. |
| `--normalize` / `--no-normalize` | Normaliza embeddings. Default: `--normalize`. |
| `--checkpoint-every` | Guarda checkpoints cada N elementos. Default: `0`. |
| `--resume` | Reanuda desde el ultimo checkpoint. |

### `src/indexing/index_builder.py`

Construye indices BM25 y/o Chroma desde un JSON con embeddings.

El indexador espera:

- `metadata.chunk_id`
- `cleaned_content`
- `embedding` para Chroma

Para BM25 solo requiere `metadata.chunk_id` y `cleaned_content`.

Corpus original:

```powershell
.\.venv\Scripts\python.exe -m src.indexing.index_builder --input-file .data\embeddings\e5_elements_with_embeddings.json --build both --bm25-index-dir .data\bm25_index --chroma-dir .data\chroma --collection-name hybridrank_elements
```

Corpus normativo:

```powershell
.\.venv\Scripts\python.exe -m src.indexing.index_builder --input-file .data\embeddings\e5_norma_fragments_with_embeddings.json --build both --bm25-index-dir .data\bm25_norma_index --chroma-dir .data\chroma_normas --collection-name hybridrank_normas
```

Argumentos:

| Argumento | Descripcion |
| --- | --- |
| `--input-file` | JSON con elementos y embeddings. Default: `.data/embeddings/e5_elements_with_embeddings.json`. |
| `--build` | `bm25`, `chroma` o `both`. Default: `chroma`. |
| `--bm25-index-dir` | Directorio de salida del indice BM25. |
| `--chroma-dir` | Directorio persistente de ChromaDB. |
| `--collection-name` | Nombre de la coleccion Chroma. |
| `--batch-size` | Batch de lectura/insercion. Default: `5000`. |
| `--tokenize-batch-size` | Batch interno de spaCy para BM25. Default: `1000`. |
| `--n-process` | Procesos spaCy para BM25. Default: `1`. |

### `src/retriever/bm25_retriever.py`

Carga un indice BM25 persistido y devuelve pares `(doc_id, score)`.

Por defecto usa `.data/bm25_index`. Para normas debe inicializarse con:

```python
BM25Retriever(index_dir=".data/bm25_norma_index")
```

### `src/retriever/dense_retriever.py`

Consulta ChromaDB con embeddings E5 de query. Devuelve pares `(doc_id, similarity)`.

Por defecto usa `.data/chroma` y `hybridrank_elements`. Para normas:

```python
DenseRetriever(
    chroma_dir=".data/chroma_normas",
    collection_name="hybridrank_normas",
)
```

### `scripts/create_evaluation_dataset.py`

Crea un dataset de evaluacion interactivo. Para cada query:

1. Ejecuta BM25 y Dense.
2. Une ambos resultados en un pool.
3. Recupera previews desde Chroma.
4. Permite marcar documentos relevantes.
5. Guarda `pool_docs` y `relevant_docs`.

En `--corpus normas`, ademas guarda:

- `pool_normas`
- `relevant_normas`
- `doc_metadata`

Formato de queries:

```json
[
  {
    "query_id": "q1",
    "query_type": "literal",
    "query": "licencia de maternidad cuba"
  }
]
```

Corpus original:

```powershell
.\.venv\Scripts\python.exe scripts\create_evaluation_dataset.py --queries scripts\sample_queries.json --corpus elements --top-k 10
```

Corpus normativo:

```powershell
.\.venv\Scripts\python.exe scripts\create_evaluation_dataset.py --queries scripts\sample_queries.json --corpus normas --top-k 10 --paginated
```

Argumentos:

| Argumento | Descripcion |
| --- | --- |
| `--queries` | JSON de queries. Obligatorio. |
| `--output` | JSON de salida. Default depende del corpus. |
| `--corpus` | `elements` o `normas`. Default: `elements`. |
| `--top-k` | Cantidad de documentos por retriever. Default: `10`. |
| `--paginated` | Modo interactivo documento por documento, con texto completo. |
| `--bm25-index-dir` | Directorio BM25 explicito. Si se omite, depende del corpus. |
| `--chroma-dir` | Directorio Chroma explicito. Si se omite, depende del corpus. |
| `--collection-name` | Coleccion Chroma explicita. Si se omite, depende del corpus. |

Defaults por corpus:

| Corpus | Output | BM25 | Chroma | Coleccion |
| --- | --- | --- | --- | --- |
| `elements` | `.data/evaluation/qrels.json` | `.data/bm25_index` | `.data/chroma` | `hybridrank_elements` |
| `normas` | `.data/evaluation/norma_qrels.json` | `.data/bm25_norma_index` | `.data/chroma_normas` | `hybridrank_normas` |

## RAG: Generacion Aumentada por Recuperacion

El modulo `src/rag/` cierra el ciclo RAG: toma los resultados de retrieval, resuelve los fragmentos normativos, construye contexto, genera una respuesta con un LLM y formatea las citas juridicas.

### Arquitectura del Pipeline

```
User Query
  |
  v
HybridRetriever (BM25 + Dense + FusionStrategy)
  |
  v
List[(fragment_id, score)]
  |
  v
NormaStore  -->  ChromaDB (.data/chroma_normas)
  |
  v
List[RetrievedFragment]  (texto + metadata juridica completa)
  |
  v
ContextBuilder  -->  bloque [Fuente 1] ... [Fuente N]
  |
  v
PromptBuilder  -->  system prompt legal + user prompt con contexto
  |
  v
GeneratorProvider  -->  Mistral AI / LiteLLM (OpenRouter, etc.)
  |
  v
CitationFormatter  -->  respuesta + bibliografia verificada
  |
  v
RAGResult (answer, sources, timings, usage)
```

### Componentes del modulo RAG

| Archivo | Responsabilidad |
| --- | --- |
| `src/rag/store/models.py` | `RetrievedFragment`: dataclass con accessors para metadata juridica. |
| `src/rag/store/norma_store.py` | `NormaStore`: resuelve `fragment_id` a texto completo via ChromaDB `.get()`. |
| `src/rag/context/context_builder.py` | `ContextBuilder`: arma bloque de contexto con headers de metadata, limites de chars/fragmentos. |
| `src/rag/prompt/prompt_builder.py` | `PromptBuilder`: genera mensajes `system`/`user` para APIs de chat. |
| `src/rag/prompt/templates.py` | Prompts en espanol especializados en legislacion cubana. |
| `src/rag/generator/base.py` | `GeneratorProvider` ABC + `GenerationResult` dataclass. |
| `src/rag/generator/mistral_provider.py` | `MistralProvider`: usa la API oficial de Mistral AI. |
| `src/rag/generator/litellm_provider.py` | `LiteLLMProvider`: soporta OpenRouter y cualquier backend compatible con LiteLLM. |
| `src/rag/generator/registry.py` | `get_generator(name)`: factory pattern, igual que `get_fusion_strategy`. |
| `src/rag/citation/citation_formatter.py` | `CitationFormatter`: extrae `[Fuente N]` del texto, genera bibliografia, detecta citas invalidas. |
| `src/rag/pipeline.py` | `RAGPipeline`: orquesta el flujo completo. `RAGResult` con respuesta, fuentes, tiempos y uso de tokens. |

### Configuracion

Copiar `.env.example` a `.env` y configurar las API keys:

```bash
cp .env.example .env
```

Variables principales:

| Variable | Descripcion | Default |
| --- | --- | --- |
| `MISTRAL_API_KEY` | API key de Mistral AI. | — |
| `OPENROUTER_API_KEY` | API key de OpenRouter (para LiteLLM). | — |
| `GENERATOR_PROVIDER` | `mistral` o `litellm`. | `mistral` |
| `GENERATOR_MODEL` | Modelo a usar. | `mistral-small-latest` |
| `GENERATOR_TEMPERATURE` | Temperatura de generacion. | `0.1` |
| `GENERATOR_MAX_TOKENS` | Maximo de tokens de respuesta. | `2048` |
| `CONTEXT_MAX_FRAGMENTS` | Maximo de fragmentos en contexto. | `8` |
| `CONTEXT_MAX_CHARS` | Limite de caracteres del contexto. | `12000` |
| `TOP_K` | Documentos recuperados. | `10` |
| `CANDIDATE_K` | Candidatos por retriever antes de fusion. | `50` |
| `FUSION_STRATEGY` | Estrategia de fusion (`rrf`, `weighted`, `hybridrank`, etc.). | `hybridrank` |

### Proveedores de Generacion

**Mistral AI** (proveedor por defecto):

```bash
GENERATOR_PROVIDER=mistral
GENERATOR_MODEL=mistral-small-latest
MISTRAL_API_KEY=tu_api_key
```

**OpenRouter via LiteLLM**:

```bash
GENERATOR_PROVIDER=litellm
GENERATOR_MODEL=openrouter/mistralai/mistral-small-3.1-24b-instruct
OPENROUTER_API_KEY=tu_api_key
```

Otros modelos compatibles con LiteLLM:

```bash
GENERATOR_MODEL=openrouter/meta-llama/llama-3.1-8b-instruct
GENERATOR_MODEL=openrouter/qwen/qwen-2.5-7b-instruct
```

### Uso programatico

```python
from src.retriever import BM25Retriever, DenseRetriever, HybridRetriever
from src.retriever.fusion import get_fusion_strategy
from src.rag import RAGPipeline, NormaStore, ContextBuilder, get_generator

# Retriever hibrido sobre corpus normativo
bm25 = BM25Retriever(index_dir=".data/bm25_norma_index")
dense = DenseRetriever(
    chroma_dir=".data/chroma_normas",
    collection_name="hybridrank_normas",
)
fusion = get_fusion_strategy("hybridrank", alpha=0.5, beta=0.5)
retriever = HybridRetriever(
    retrievers={"bm25": bm25, "dense": dense},
    fusion_strategy=fusion,
    candidate_k=50,
)

# Pipeline RAG
pipeline = RAGPipeline(
    retriever=retriever,
    generator=get_generator(),  # usa GENERATOR_PROVIDER del .env
    top_k=10,
)

result = pipeline.run("Que establece el Decreto-Ley 114 de 2025?")
print(result.answer)       # respuesta con citas [Fuente N] + bibliografia
print(result.fragments)    # fragmentos recuperados con metadata
print(result.total_time_ms)
```

El pipeline es desacoplado: acepta cualquier `Retriever` (BM25, Dense o Hybrid), cualquier `GeneratorProvider` (Mistral, LiteLLM), y los componentes intermedios son configurables.

### Demo con Chainlit

La aplicacion `app/chainlit_app.py` ofrece una interfaz web interactiva para consultar el sistema RAG.

#### Prerequisitos

1. Indices BM25 y Chroma normativos construidos (ver [Flujo Recomendado Para Normas](#flujo-recomendado-para-normas), pasos 1-4).
2. Archivo `.env` configurado con al menos una API key de generacion (`MISTRAL_API_KEY` o `OPENROUTER_API_KEY`).

#### Instalacion y ejecucion

```powershell
.\.venv\Scripts\pip.exe install chainlit
.\.venv\Scripts\chainlit.exe run app\chainlit_app.py
```

Chainlit abre automaticamente `http://localhost:8000` en el navegador.

#### Flujo de la aplicacion

**Al iniciar la sesion** (`on_chat_start`):

1. Construye un `BM25Retriever` con el indice normativo (`.data/bm25_norma_index`).
2. Construye un `DenseRetriever` con ChromaDB normativo (`.data/chroma_normas`, coleccion `hybridrank_normas`).
3. Crea un `HybridRetriever` combinando ambos con la `FusionStrategy` configurada en `.env` (default: `hybridrank`).
4. Inicializa `NormaStore`, `ContextBuilder` y `GeneratorProvider` segun la configuracion.
5. Muestra un mensaje de bienvenida con la configuracion activa: retriever, generador, top-k y estrategia de fusion.

**Al recibir una consulta** (`on_message`):

1. **Retrieval**: ejecuta `pipeline.run(query)`, que internamente consulta BM25 y Dense por separado, fusiona los rankings, y resuelve los `fragment_id` a texto completo via `NormaStore`.
2. **Generacion**: construye el contexto con headers `[Fuente N]` y metadata juridica, arma los mensajes con el prompt legal, y envia al LLM configurado.
3. **Respuesta**: muestra el texto generado con citas `[Fuente N]` seguido de una seccion "Fuentes Consultadas" con bibliografia verificada.
4. **Metadata de ejecucion**: muestra tiempos de retrieval, generacion y total en milisegundos, mas el desglose de tokens (prompt + completion = total).
5. **Fragmentos en panel lateral**: presenta hasta 5 fragmentos recuperados como elementos expandibles, cada uno con `citation_key` (ej. `Decreto-Ley 114/2025 (GOC-2026-215-O24)`), organismo emisor, score de relevancia, `fragment_id`, y un snippet de hasta 600 caracteres del contenido.

#### Configuracion

El retriever, la estrategia de fusion, el proveedor de generacion y el modelo se controlan via `.env`. Cambiar una variable y reiniciar Chainlit aplica la nueva configuracion sin modificar codigo.

### Logging de Interacciones

Cada consulta se registra automaticamente en `.data/logs/rag_interactions.jsonl`:

```json
{
  "timestamp": "2026-05-15T...",
  "query": "...",
  "retriever": "HybridRetriever(bm25+dense|HybridRankFusion)",
  "provider": "mistral-small-latest",
  "retrieved_docs": [{"fragment_id": "...", "score": 0.842}],
  "sources": ["Decreto-Ley 114/2025 (GOC-2026-215-O24)"],
  "answer": "...",
  "usage": {"prompt_tokens": 1200, "completion_tokens": 400, "total_tokens": 1600},
  "retrieval_time_ms": 120.5,
  "generation_time_ms": 2340.1,
  "total_time_ms": 2461.8
}
```

## Flujo Recomendado Para Normas

1. Procesar normas desde elementos limpios:

```powershell
.\.venv\Scripts\python.exe -m src.data_preparation.norma_processor
```

2. Exportar fragmentos normativos:

```powershell
.\.venv\Scripts\python.exe -m src.data_preparation.norma_index_document_builder
```

3. Generar embeddings:

```powershell
.\.venv\Scripts\python.exe src\embedding_model\embedding_generator_e5.py --input-file .data\norma_output\norma_fragments.json --output-file .data\embeddings\e5_norma_fragments_with_embeddings.json --stats-file .data\embeddings\e5_norma_embedding_stats.json
```

4. Construir BM25 y Chroma normativos:

```powershell
.\.venv\Scripts\python.exe -m src.indexing.index_builder --input-file .data\embeddings\e5_norma_fragments_with_embeddings.json --build both --bm25-index-dir .data\bm25_norma_index --chroma-dir .data\chroma_normas --collection-name hybridrank_normas
```

5. Crear dataset de evaluacion normativo:

```powershell
.\.venv\Scripts\python.exe scripts\create_evaluation_dataset.py --queries scripts\sample_queries.json --corpus normas --top-k 10 --paginated
```

## Verificacion

Tests focalizados:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_norma_index_document_builder.py tests/test_index_builder.py tests/test_create_evaluation_dataset_normas.py -q
```

Suite relevante de normas, indices y retrievers:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_norma_models.py tests/test_norma_processor.py tests/test_retrievers.py tests/test_norma_index_document_builder.py tests/test_index_builder.py tests/test_create_evaluation_dataset_normas.py -q
```

## Notas

- `.data/` esta ignorado por git; los indices, embeddings y datasets generados son artifacts locales.
- El corpus normativo evalua relevancia a nivel de fragmento, pero conserva `norma_id` para agrupar despues por norma.
- Los segmentos `unmatched` del procesamiento de normas no se indexan en la version actual porque no tienen identidad normativa confiable.
