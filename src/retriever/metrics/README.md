# Métricas de Evaluación para Sistemas RAG

Este módulo implementa las métricas clásicas de evaluación para sistemas de recuperación de información (Information Retrieval), adaptadas para evaluar sistemas RAG (Retrieval-Augmented Generation).

## Métricas Implementadas

### 1. **Recall@k**
Mide la proporción de documentos relevantes que aparecen en los top-k documentos recuperados.

**Fórmula:**
```
Recall@k = (# docs relevantes en top-k) / (# docs relevantes totales)
```

**Uso en RAG:** Recall@k mide la probabilidad de que al menos un pasaje útil esté disponible para el LLM. Un Recall@k alto asegura que el modelo tenga acceso a la información necesaria para generar respuestas precisas.

### 2. **Precision@k**
Mide la proporción de documentos relevantes dentro de los top-k documentos recuperados.

**Fórmula:**
```
Precision@k = (# docs relevantes en top-k) / k
```

**Uso en RAG:** Precision@k mide la "limpieza" o "pureza" del ranking. Un Precision@k alto significa que el sistema evita incluir documentos irrelevantes en el top-k, reduciendo el ruido en el contexto proporcionado al LLM y evitando confusión que podría llevar a respuestas incorrectas.

**Diferencia con Recall:** Mientras Recall mide cobertura (¿encontramos lo relevante?), Precision mide limpieza (¿evitamos lo irrelevante?).

### 3. **F1@k**
Media armónica entre Precision@k y Recall@k que balancea ambas métricas.

**Fórmula:**
```
F1@k = 2 * (Precision@k * Recall@k) / (Precision@k + Recall@k)
```

**Uso en RAG:** F1@k proporciona un balance óptimo entre dar suficiente contexto al LLM (Recall) y evitar ruido (Precision). Es especialmente útil para encontrar el valor óptimo de k que maximiza la calidad del retrieval. Un F1@k alto indica que el sistema logra alta cobertura sin sacrificar precisión.

### 4. **MRR (Mean Reciprocal Rank)**
Mide la posición del primer documento relevante en el ranking.

**Fórmula:**
```
MRR = 1 / rank_del_primer_documento_relevante
```

**Uso en RAG:** MRR alto indica que documentos relevantes aparecen temprano en el ranking, reduciendo la "distracción" del LLM con contenido irrelevante al principio del contexto.

### 5. **MAP (Mean Average Precision)**
Promedio de las precisiones en cada posición donde aparece un documento relevante.

**Fórmula:**
```
AP = (1/|relevantes|) × Σ(Precision@i × rel(i))
```

**Uso en RAG:** MAP evalúa la calidad general del ranking. Un MAP alto significa que el sistema mantiene buena precisión a lo largo de todo el ranking, no solo al principio.

### 6. **nDCG@k (Normalized Discounted Cumulative Gain)**
Mide la calidad del ranking considerando tanto relevancia como posición, con descuento logarítmico.

**Fórmula:**
```
DCG@k = Σ(i=1 to k) [(2^rel_i - 1) / log2(i + 1)]
nDCG@k = DCG@k / IDCG@k
```

**Uso en RAG:** nDCG penaliza documentos relevantes que aparecen tarde en el ranking. Valores altos indican que los documentos más relevantes están bien posicionados.

## Instalación

Las dependencias necesarias están en `requirements.txt`:
```bash
pip install -r requirements.txt
```

Dependencias clave:
- `numpy` - Para cálculos numéricos (usado en nDCG)
- `typing` - Type hints (incluido en Python 3.5+)

## Uso Básico

```python
from src.retriever.metrics import RecallAtK, PrecisionAtK, F1AtK, MRR, MAP, NDCG

# Documentos recuperados: lista de tuplas (doc_id, score)
# Ordenados por relevancia descendente
retrieved_documents = [
    ("doc1", 0.95),
    ("doc2", 0.87),
    ("doc3", 0.76),
    ("doc4", 0.68),
]

# Documentos que son realmente relevantes
relevant_documents = ["doc1", "doc3", "doc5"]

# Calcular Recall@3
recall = RecallAtK()
result = recall.compute(retrieved_documents, relevant_documents, k=3)
print(f"Recall@3: {result['score']:.3f}")
print(f"Encontrados: {result['relevant_found']}/{result['total_relevant']}")

# Calcular Precision@3
precision = PrecisionAtK()
result = precision.compute(retrieved_documents, relevant_documents, k=3)
print(f"Precision@3: {result['score']:.3f}")
print(f"Relevantes de {result['total_retrieved']}: {result['relevant_found']}")

# Calcular F1@3
f1 = F1AtK()
result = f1.compute(retrieved_documents, relevant_documents, k=3)
print(f"F1@3: {result['score']:.3f}")
print(f"P={result['precision']:.3f}, R={result['recall']:.3f}")

# Calcular MRR
mrr = MRR()
result = mrr.compute(retrieved_documents, relevant_documents)
print(f"MRR: {result['score']:.3f}")
print(f"Primera posición relevante: {result['first_relevant_rank']}")

# Calcular MAP
map_metric = MAP()
result = map_metric.compute(retrieved_documents, relevant_documents)
print(f"MAP: {result['score']:.3f}")

# Calcular nDCG@5
ndcg = NDCG()
result = ndcg.compute(retrieved_documents, relevant_documents, k=5)
print(f"nDCG@5: {result['score']:.3f}")
```

## Formato de Entrada/Salida

### Entrada

Todas las métricas reciben:

- **`retrieved_documents`**: `List[Tuple[str, float]]`
  - Lista de tuplas `(doc_id, score)` ordenadas por relevancia descendente
  - `doc_id` es el identificador único del documento (ej: `chunk_id`)
  - `score` es la puntuación de relevancia del sistema

- **`relevant_documents`**: `List[str]`
  - Lista de `doc_id` que son relevantes para la consulta
  - Típicamente obtenidos de anotaciones manuales o ground truth

- **`k`**: `Optional[int]`
  - Número de documentos top a considerar
  - Si es `None`, usa todos los documentos recuperados

### Salida

Todas las métricas retornan un `Dict[str, Any]` con al menos:

```python
{
    'score': float,        # Valor numérico de la métrica [0, 1]
    'metric_name': str,    # Nombre descriptivo (ej: "Recall@5", "MRR")
    # ... metadata adicional específica de cada métrica
}
```

#### Metadata específica:

**RecallAtK:**
```python
{
    'score': 0.667,
    'metric_name': 'Recall@3',
    'k': 3,
    'relevant_found': 2,
    'total_relevant': 3
}
```

**PrecisionAtK:**
```python
{
    'score': 0.667,
    'metric_name': 'Precision@3',
    'k': 3,
    'relevant_found': 2,
    'total_retrieved': 3
}
```

**F1AtK:**
```python
{
    'score': 0.667,
    'metric_name': 'F1@3',
    'k': 3,
    'precision': 0.667,
    'recall': 0.667,
    'relevant_found': 2,
    'total_relevant': 3,
    'total_retrieved': 3
}
```

**MRR:**
```python
{
    'score': 0.5,
    'metric_name': 'MRR',
    'first_relevant_rank': 2  # o None si no hay relevantes
}
```

**MAP:**
```python
{
    'score': 0.583,
    'metric_name': 'MAP',
    'average_precision': 0.583,
    'precisions_at_relevant': [0.5, 0.667],
    'num_relevant_retrieved': 2
}
```

**NDCG:**
```python
{
    'score': 0.704,
    'metric_name': 'nDCG@3',
    'k': 3,
    'dcg': 1.5,
    'idcg': 2.131
}
```

## Ejemplo Completo

Ver [`examples/metrics_usage.py`](../examples/metrics_usage.py) para un ejemplo completo con interpretación de resultados.

```bash
python examples/metrics_usage.py
```

## Tests

El módulo incluye una suite completa de tests unitarios:

```bash
# Ejecutar todos los tests
python -m unittest tests.test_metrics -v

# Ejecutar tests de una métrica específica
python -m unittest tests.test_metrics.TestRecallAtK -v
```

Coverage: 39 tests cubriendo:
- Casos básicos de cada métrica (Recall, Precision, F1, MRR, MAP, nDCG)
- Casos edge: listas vacías, sin relevantes, ranking perfecto
- Parámetro k opcional
- Verificación de fórmulas (especialmente F1 con respecto a P y R)
- Interfaz consistente entre todas las métricas

## Interpretación para RAG

### Valores recomendados:

| Métrica | Bueno | Excelente | Impacto en RAG |
|---------|-------|-----------|----------------|
| Recall@k | >0.6 | >0.8 | Cobertura de información relevante |
| Precision@k | >0.6 | >0.8 | Limpieza del contexto (bajo ruido) |
| F1@k | >0.6 | >0.8 | Balance óptimo cobertura-limpieza |
| MRR | >0.5 | >0.7 | Relevancia temprana reduce hallucinations |
| MAP | >0.5 | >0.7 | Calidad general del contexto |
| nDCG@k | >0.6 | >0.8 | Orden óptimo para el LLM |

### Relación con hallucinations:

- **Recall@k bajo** → LLM no tiene acceso a info relevante → Mayor riesgo de hallucinations
- **Precision@k bajo** → Mucho ruido en el contexto → LLM se confunde con información irrelevante
- **F1@k bajo** → Desbalance entre cobertura y limpieza → Contexto suboptimal
- **MRR/nDCG bajos** → Contexto irrelevante primero → LLM puede confundirse
- **MAP bajo** → Ranking de baja calidad → Respuestas menos precisas

### Trade-offs:

- **Aumentar k**: Mejora Recall pero puede degradar Precision → Más cobertura pero más ruido
- **Disminuir k**: Mejora Precision pero puede degradar Recall → Menos ruido pero menos cobertura
- **k óptimo**: Maximiza F1@k, balanceando ambas métricas según la aplicación
- **Reranking**: Mejora nDCG/MAP/Precision pero añade latencia
- **Hybrid retrieval**: Puede mejorar todas las métricas pero aumenta costo computacional

### Ejemplo de análisis:

Si observas:
- Recall@5 = 0.8, Precision@5 = 0.4, F1@5 = 0.53
- Recall@3 = 0.6, Precision@3 = 0.7, F1@3 = 0.65

**Conclusión**: k=3 es mejor que k=5 (mayor F1), ya que el aumento en Recall no compensa la pérdida de Precision.

## Implementación de Nuevas Métricas

Para añadir una nueva métrica, hereda de la clase `Metric` base:

```python
from typing import List, Tuple, Dict, Any, Optional
from .base import Metric

class MyMetric(Metric):
    def compute(
        self,
        retrieved_documents: List[Tuple[str, float]],
        relevant_documents: List[str],
        k: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        # Tu implementación aquí
        score = ...
        
        return {
            'score': score,
            'metric_name': 'MyMetric',
            # ... metadata adicional
        }
    
    @property
    def name(self) -> str:
        return "MyMetric"
```

## Referencias

1. [Relevance, Precision, and Recall](https://course.khoury.northeastern.edu/cs6200sp15/slides/m06.s02%20-%20relevance,%20precision,%20and%20recall.pdf)
2. [Offline Evaluation - Pinecone](https://www.pinecone.io/learn/offline-evaluation/)
3. [nDCG Metric - Evidently AI](https://www.evidentlyai.com/ranking-metrics/ndcg-metric)
4. [RAG Evaluation - Vectara](https://www.vectara.com/blog/evaluating-rag)
5. [RAGAs Framework](https://arxiv.org/html/2505.04847v2)

## Licencia

Parte del proyecto HybridRank - Sistema RAG para recuperación de información.
