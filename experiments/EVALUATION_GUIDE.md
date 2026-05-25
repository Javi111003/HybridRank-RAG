# Guia de Evaluacion de Estrategias de Fusion

Documentacion completa del script `evaluate_fusion_strategies.py`: arquitectura, configuracion,
parametros modificables, ejemplos de uso y direcciones de mejora.

---

## 1. Arquitectura General

```
evaluate_fusion_strategies.py
    |
    |-- load_evaluation_data()        -> Lee norma_qrels.json
    |-- define_strategies()           -> Configura que estrategias evaluar
    |-- build_retriever(config)       -> Instancia el retriever apropiado
    |-- evaluate_single_query(...)    -> Ejecuta metricas sobre una query
    |-- run_evaluation()              -> Orquesta todo y genera CSVs
```

**Flujo de ejecucion:**

1. Carga las queries de evaluacion con sus documentos relevantes (qrels)
2. Inicializa BM25Retriever y DenseRetriever sobre los indices de normas
3. Para cada estrategia configurada, construye un HybridRetriever con esa fusion
4. Evalua cada query con todas las metricas en todos los valores de k
5. Genera `fusion_metrics.csv` (detallado) y `fusion_summary.csv` (agregado)

---

## 2. Archivos de Entrada y Salida

| Archivo | Rol |
|---------|-----|
| `.data/evaluation/norma_qrels.json` | Dataset de evaluacion (queries + relevancia) |
| `.data/bm25_norma_index/` | Indice BM25 para fragmentos de normas |
| `.data/chroma_normas/` | Indice denso ChromaDB para fragmentos de normas |
| `experiments/results/fusion_metrics.csv` | Resultados detallados (1 fila por query/k/estrategia) |
| `experiments/results/fusion_summary.csv` | Promedios por estrategia |

---

## 3. Parametros Globales Modificables

En la seccion de constantes del script:

```python
EVALUATION_DATA_PATH = project_root / ".data" / "evaluation" / "norma_qrels.json"
OUTPUT_DIR = project_root / "experiments" / "results"
K_VALUES = [5, 10, 20]
```

### Modificar valores de k

Agregar o cambiar los puntos de corte de evaluacion:

```python
# Evaluacion mas granular
K_VALUES = [1, 3, 5, 10, 15, 20, 50]

# Solo top-10
K_VALUES = [10]
```

### Modificar candidate_k

En `build_retriever()`, el parametro `candidate_k=50` controla cuantos documentos
recupera cada retriever ANTES de la fusion. Debe ser >= max(K_VALUES):

```python
# Mas candidatos = mas documentos para fusionar (potencialmente mejor recall)
hybrid = HybridRetriever(
    retrievers={"bm25": bm25, "dense": dense},
    fusion_strategy=fusion_strategy,
    candidate_k=100,  # default: 50
)
```

---

## 4. Estrategias Disponibles

### 4.1 Baselines

```python
"bm25_only": {"type": "baseline", "retriever": "bm25"}
"dense_only": {"type": "baseline", "retriever": "dense"}
```

No usan fusion. Ejecutan un solo retriever directamente.

---

### 4.2 RRF (Reciprocal Rank Fusion)

**Tipo:** `"rrf"` | **Familia:** Basada en rango

| Parametro | Tipo | Default | Descripcion |
|-----------|------|---------|-------------|
| `k` | int | 60 | Suavizado. Mayor k = menos peso a posiciones altas |

```python
"rrf_k60": {"type": "rrf", "params": {"k": 60}}
"rrf_k30": {"type": "rrf", "params": {"k": 30}}   # mas agresivo con top ranks
"rrf_k200": {"type": "rrf", "params": {"k": 200}}  # mas uniforme
```

**Formula:** `score(d) = sum_retrievers(1 / (k + rank_i(d)))`

**Cuando usar:** Cuando no se confian en los scores absolutos de los retrievers
y solo importan las posiciones relativas.

---

### 4.3 Borda Count

**Tipo:** `"borda"` | **Familia:** Basada en rango

Sin parametros.

```python
"borda": {"type": "borda", "params": {}}
```

**Formula:** `score(d) = sum_retrievers(N_i - rank_i(d) + 1)` donde N_i = total de documentos del retriever i.

**Cuando usar:** Alternativa simple a RRF. Asigna mas puntos proporcionalmente al tamano del ranking.

---

### 4.4 CombSUM

**Tipo:** `"combsum"` | **Familia:** Basada en scores

| Parametro | Tipo | Default | Opciones |
|-----------|------|---------|----------|
| `normalizer` | str | "minmax" | "minmax", "zscore", "sum", "identity" |

```python
"combsum_minmax": {"type": "combsum", "params": {"normalizer": "minmax"}}
"combsum_zscore": {"type": "combsum", "params": {"normalizer": "zscore"}}
"combsum_sum":    {"type": "combsum", "params": {"normalizer": "sum"}}
"combsum_raw":    {"type": "combsum", "params": {"normalizer": "identity"}}
```

**Formula:** `score(d) = sum_retrievers(norm_score_i(d))`

**Cuando usar:** Cuando los scores de los retrievers son comparables despues de normalizacion.

---

### 4.5 CombMNZ

**Tipo:** `"combmnz"` | **Familia:** Basada en scores

| Parametro | Tipo | Default | Opciones |
|-----------|------|---------|----------|
| `normalizer` | str | "minmax" | "minmax", "zscore", "sum", "identity" |

```python
"combmnz_minmax":  {"type": "combmnz", "params": {"normalizer": "minmax"}}
"combmnz_zscore":  {"type": "combmnz", "params": {"normalizer": "zscore"}}
```

**Formula:** `score(d) = count_nonzero(d) * sum_retrievers(norm_score_i(d))`

**Cuando usar:** Cuando se quiere premiar el consenso entre retrievers. Un documento
que aparece en ambos rankings recibe un boost multiplicativo.

---

### 4.6 Weighted Score Fusion

**Tipo:** `"weighted"` | **Familia:** Basada en scores

| Parametro | Tipo | Default | Descripcion |
|-----------|------|---------|-------------|
| `alpha` | float | 0.5 | Peso para BM25. Dense recibe (1-alpha) |
| `sparse_key` | str | "bm25" | Clave del retriever disperso |
| `dense_key` | str | "dense" | Clave del retriever denso |
| `normalizer` | str | "minmax" | Normalizador de scores |

```python
"weighted_a0.3": {"type": "weighted", "params": {"alpha": 0.3}}  # favorece dense
"weighted_a0.5": {"type": "weighted", "params": {"alpha": 0.5}}  # equilibrado
"weighted_a0.7": {"type": "weighted", "params": {"alpha": 0.7}}  # favorece BM25
"weighted_a0.9": {"type": "weighted", "params": {"alpha": 0.9}}  # casi solo BM25
```

**Formula:** `score(d) = alpha * norm_bm25(d) + (1 - alpha) * norm_dense(d)`

**Cuando usar:** Control directo sobre la importancia relativa de cada retriever.

---

### 4.7 HybridRank (Propuesta Original)

**Tipo:** `"hybridrank"` | **Familia:** Hibrida (rango + score)

| Parametro | Tipo | Default | Descripcion |
|-----------|------|---------|-------------|
| `alpha` | float | 0.5 | Peso BM25 en componente weighted |
| `beta` | float | 0.5 | Peso del componente RRF en fusion final |
| `k` | int | 60 | Parametro de suavizado RRF |
| `sparse_key` | str | "bm25" | Clave del retriever disperso |
| `dense_key` | str | "dense" | Clave del retriever denso |
| `normalizer` | str | "minmax" | Normalizador para scores weighted |
| `rrf_normalizer` | str | "minmax" | Normalizador para scores RRF |

```python
# Exploracion completa del espacio de parametros
"hybridrank_a0.7_b0.3": {"type": "hybridrank", "params": {"alpha": 0.7, "beta": 0.3}}
"hybridrank_a0.7_b0.3_k30": {"type": "hybridrank", "params": {"alpha": 0.7, "beta": 0.3, "k": 30}}
"hybridrank_a0.5_b0.5_zscore": {"type": "hybridrank", "params": {
    "alpha": 0.5, "beta": 0.5, "normalizer": "zscore", "rrf_normalizer": "zscore"
}}
```

**Formula (2 niveles):**
1. `weighted(d) = alpha * norm_bm25(d) + (1-alpha) * norm_dense(d)`
2. `rrf(d) = sum_retrievers(1 / (k + rank_i(d)))` -> normalizado
3. `final(d) = beta * norm_rrf(d) + (1-beta) * weighted(d)`

**Casos especiales:**
- `beta=0`: Colapsa a WeightedScoreFusion puro
- `beta=1`: Colapsa a RRF puro
- `alpha=1, beta=0`: Solo BM25 con normalizacion

---

## 5. Normalizadores de Scores

Los normalizadores transforman los scores antes de combinarlos:

| Nombre | Comportamiento | Mejor para |
|--------|---------------|------------|
| `"minmax"` | Escala a [0, 1] | Default seguro, preserva distribuciones |
| `"zscore"` | Centra en media=0, std=1 | Scores con distribuciones gaussianas |
| `"sum"` | Divide por la suma total | Rankings cortos, scores no-negativos |
| `"identity"` | No transforma | Cuando los scores ya son comparables |

---

## 6. Metricas de Evaluacion

| Metrica | Nombre CSV | Que mide |
|---------|-----------|----------|
| RecallAtK | `recall` | Proporcion de relevantes encontrados en top-k |
| PrecisionAtK | `precision` | Proporcion de top-k que son relevantes |
| F1AtK | `f1` | Media armonica de precision y recall |
| MRR | `mrr` | Posicion del primer documento relevante |
| MAP | `map` | Precision promediada en posiciones relevantes |
| NDCG | `ndcg` | Ganancia acumulada descontada normalizada |

---

## 7. Formato del Dataset de Evaluacion (qrels)

Archivo: `.data/evaluation/norma_qrels.json`

```json
[
  {
    "query_id": "q1",
    "query_type": "referencia_exacta",
    "query": "Que establece el Decreto-Ley 114 del 2025 sobre...",
    "pool_docs": ["decreto_ley_114_2025_...__f000", "decreto_ley_114_2025_...__f001", ...],
    "relevant_docs": ["decreto_ley_114_2025_...__f003", "decreto_ley_114_2025_...__f005"],
    "pool_normas": ["decreto_ley_114_2025_consejo_de_estado"],
    "relevant_normas": ["decreto_ley_114_2025_consejo_de_estado"],
    "doc_metadata": { ... }
  }
]
```

**Campos clave:**
- `relevant_docs`: Los IDs de fragmentos que se consideran respuesta correcta
- `query_type`: Permite analisis segmentado (referencia_exacta, semantica, etc.)

---

## 8. Ejemplos de Uso

### Ejecutar evaluacion completa

```bash
.venv/Scripts/python.exe experiments/evaluate_fusion_strategies.py
```

### Agregar una nueva estrategia

En `define_strategies()`, agregar una entrada al diccionario:

```python
# Nueva variante RRF con k=10
"rrf_k10": {"type": "rrf", "params": {"k": 10}},

# CombMNZ con z-score
"combmnz_zscore": {"type": "combmnz", "params": {"normalizer": "zscore"}},

# HybridRank con parametros extremos
"hybridrank_aggressive": {"type": "hybridrank", "params": {
    "alpha": 0.9, "beta": 0.1, "k": 30, "normalizer": "zscore"
}},
```

### Evaluar solo un subconjunto

Modificar temporalmente `define_strategies()`:

```python
def define_strategies():
    return {
        "bm25_only": {"type": "baseline", "retriever": "bm25"},
        "hybridrank_best": {"type": "hybridrank", "params": {"alpha": 0.7, "beta": 0.3}},
    }
```

### Cambiar el dataset de evaluacion

```python
EVALUATION_DATA_PATH = project_root / ".data" / "evaluation" / "mi_nuevo_qrels.json"
```

### Grid search manual de parametros HybridRank

```python
def define_strategies():
    strategies = {
        "bm25_only": {"type": "baseline", "retriever": "bm25"},
        "dense_only": {"type": "baseline", "retriever": "dense"},
    }
    # Grid sobre alpha y beta
    for alpha in [0.3, 0.5, 0.7, 0.9]:
        for beta in [0.1, 0.3, 0.5, 0.7]:
            name = f"hr_a{alpha}_b{beta}"
            strategies[name] = {
                "type": "hybridrank",
                "params": {"alpha": alpha, "beta": beta}
            }
    return strategies
```

---

## 9. Estructura del CSV de Salida

`fusion_metrics.csv` tiene una fila por cada combinacion (estrategia, query, k):

| Columna | Descripcion |
|---------|-------------|
| `top_k` | Valor de k para esta fila (5, 10, 20) |
| `recall` | Recall@k |
| `precision` | Precision@k |
| `f1` | F1@k |
| `mrr` | Mean Reciprocal Rank |
| `map` | Mean Average Precision@k |
| `ndcg` | nDCG@k |
| `strategy` | Nombre de la estrategia |
| `query_id` | ID de la query evaluada |
| `query_type` | Tipo de query |
| `retriever_name` | Nombre completo del retriever |

Total de filas: `num_estrategias * num_queries * num_k_values`

---

## 10. Direcciones de Mejora y Exploracion

### 10.1 Ampliar el espacio de parametros

- **RRF k**: Probar k en {10, 20, 30, 40, 60, 80, 100, 150}
- **HybridRank**: Grid mas fino de alpha/beta con paso 0.1
- **Normalizadores**: Combinar diferentes normalizadores entre componentes
  (e.g., zscore para weighted + minmax para RRF)

### 10.2 Nuevas estrategias de fusion

Ideas que se pueden implementar siguiendo el patron de `FusionStrategy`:

- **CombANZ** (Average Non-Zero): Como CombSUM pero divide por count_nonzero en vez de multiplicar
- **ISR (Inverse Square Rank):** `score(d) = sum(1 / rank^2)` — penaliza mas las posiciones bajas
- **Condorcet Fusion:** Votacion por pares entre documentos
- **Linear Combination con pesos aprendidos:** Optimizar pesos via queries de entrenamiento

### 10.3 Mas retrievers

El `HybridRetriever` acepta N retrievers. Se podria agregar:

```python
retrievers = {
    "bm25": bm25,
    "dense": dense,
    "reranker": reranker,  # un cross-encoder como tercer retriever
}
```

Las estrategias basadas en rango (RRF, Borda, CombSUM, CombMNZ) ya soportan
N retrievers. Solo WeightedScore y HybridRank estan limitadas a 2.

### 10.4 Mejorar candidate_k dinamicamente

En vez de un candidate_k fijo, se podria:
- Usar candidate_k = 2 * top_k (adaptativo)
- Usar candidate_k diferente por retriever (BM25 mas barato, puede dar mas)

### 10.5 Evaluacion mas robusta

- **Significancia estadistica:** Agregar test de Wilcoxon entre estrategias
- **Bootstrap confidence intervals:** Para las metricas promedio
- **Cross-validation por queries:** Dividir queries en folds
- **Mas queries:** El dataset actual tiene 20 queries; ampliar a 50+ para mayor confianza

### 10.6 Metricas adicionales

Se pueden implementar nuevas metricas heredando de `Metric`:

- **P@1:** Precision solo en el primer resultado (critical para RAG)
- **R-Precision:** Precision hasta el numero de documentos relevantes
- **Reciprocal Rank @k:** MRR pero truncado a k

### 10.7 Evaluacion por dificultad de query

Segmentar resultados no solo por `query_type` sino por dificultad:
- Queries con muchos docs relevantes (faciles) vs pocos (dificiles)
- Queries donde BM25 y Dense discrepan mucho (oportunidad para fusion)

### 10.8 Optimizacion automatica

Implementar un script de hyperparameter tuning:

```python
from scipy.optimize import minimize

def objective(params):
    alpha, beta = params
    # Evaluar HybridRank con estos params
    # Retornar -ndcg promedio (minimizar negativo = maximizar)
    ...

result = minimize(objective, x0=[0.5, 0.5], bounds=[(0,1), (0,1)])
```

---

## 11. Troubleshooting

| Problema | Causa | Solucion |
|----------|-------|----------|
| Todas las metricas = 0 | IDs del indice no coinciden con qrels | Verificar que se usan indices de normas (`bm25_norma_index`, `chroma_normas`) |
| NaN en el CSV | Bug anterior donde columnas tenian @k embebido | Ya corregido: columnas son genericas (recall, precision, etc.) |
| UnicodeEncodeError | Caracteres especiales en print() bajo Windows cp1252 | Evitar emojis/flechas unicode en prints |
| Evaluacion muy lenta | DenseRetriever carga modelo en cada query | Normal en CPU; considerar GPU o reducir num_estrategias |
