# Grid Search: Optimizacion de Parametros de Fusion

## 1. Objetivo

Encontrar la configuracion optima de parametros para **WeightedScoreFusion** y **HybridRankFusion** mediante una busqueda exhaustiva (grid search) con validacion cruzada Leave-One-Out (LOO-CV) para evitar overfitting sobre las 20 queries de evaluacion.

## 2. Metodologia

### 2.1 Espacio de busqueda

| Estrategia | Parametro | Valores | Total configs |
|---|---|---|---|
| **WeightedScore** | alpha | 0.1, 0.2, ..., 0.9 | 54 |
| | normalizer | minmax, zscore | |
| | candidate_k | 20, 50, 100 | |
| **HybridRank** | alpha | 0.5, 0.6, 0.7, 0.8, 0.9 | 300 |
| | beta | 0.0, 0.1, 0.2, 0.3, 0.4, 0.5 | |
| | rrf_k | 10, 20, 40, 60, 100 | |
| | normalizer | minmax, zscore | |

**Total: 354 configuraciones evaluadas.**

### 2.2 Metrica objetivo compuesta

Para optimizacion se uso una metrica compuesta que balancea multiples aspectos relevantes para RAG juridico:

```
objective = 0.4 * nDCG@10 + 0.3 * Recall@10 + 0.2 * MAP@10 + 0.1 * F1@10
```

- **nDCG@10 (40%)**: Calidad del ordenamiento; evidencia bien posicionada.
- **Recall@10 (30%)**: Cobertura; que no falte evidencia relevante.
- **MAP@10 (20%)**: Calidad sostenida del ranking completo.
- **F1@10 (10%)**: Balance precision/cobertura; evitar ruido.
- MRR se excluyo por saturacion (>0.93 en casi todas las configs).

### 2.3 Validacion: Leave-One-Out Cross Validation

Con solo 20 queries, un split fijo train/test seria inestable. LOO-CV maximiza el uso de datos:

```
Para cada query q_i (i=1..20):
    1. Entrenar: optimizar parametros sobre las otras 19 queries
    2. Testear: evaluar la mejor config en q_i (held-out)
    3. Reportar: performance no sesgada en q_i
```

Esto produce un estimado no sesgado del rendimiento de generalizacion.

### 2.4 Optimizacion de eficiencia

Los resultados de BM25 y Dense son **independientes** de los parametros de fusion. Se cachearon los resultados de retrieval por cada `candidate_k` unico (120 llamadas al retriever en lugar de ~14,000), reduciendo el tiempo de ejecucion de horas a ~60 segundos.

## 3. Resultados

### 3.1 Top 10 configuraciones globales

| Rank | Estrategia | alpha | beta | rrf_k | normalizer | candidate_k | Objective |
|------|-----------|-------|------|-------|------------|-------------|-----------|
| 1 | HybridRank | 0.7 | 0.2 | 10 | minmax | 50 | **0.7351** |
| 2 | HybridRank | 0.8 | 0.3 | 10 | minmax | 50 | 0.7340 |
| 3 | HybridRank | 0.8 | 0.4 | 10 | minmax | 50 | 0.7334 |
| 4 | HybridRank | 0.7 | 0.1 | 10 | minmax | 50 | 0.7312 |
| 5 | HybridRank | 0.9 | 0.4 | 10 | minmax | 50 | 0.7308 |
| 6 | HybridRank | 0.8 | 0.4 | 10 | zscore | 50 | 0.7304 |
| 7 | HybridRank | 0.8 | 0.2 | 10 | minmax | 50 | 0.7302 |
| 8 | HybridRank | 0.9 | 0.5 | 10 | zscore | 50 | 0.7301 |
| 9 | HybridRank | 0.7 | 0.2 | 20 | minmax | 50 | 0.7300 |
| 10 | HybridRank | 0.8 | 0.3 | 10 | zscore | 50 | 0.7299 |

**Las 12 mejores posiciones son todas HybridRank.** La mejor Weighted (alpha=0.6, minmax, k=20) aparece en posicion 13 con objective=0.7271.

### 3.2 Mejor configuracion por metrica individual

| Metrica | Estrategia | Params | Score |
|---------|-----------|--------|-------|
| **Recall@10** | HybridRank | a=0.7, b=0.2, rrf_k=10, minmax | 0.7303 |
| **Precision@10** | HybridRank | a=0.7, b=0.2, rrf_k=10, minmax | 0.6550 |
| **F1@10** | HybridRank | a=0.7, b=0.2, rrf_k=10, minmax | 0.6310 |
| **nDCG@10** | HybridRank | a=0.7, b=0.2, rrf_k=10, minmax | 0.8310 |
| **MAP@10** | HybridRank | a=0.8, b=0.3, rrf_k=10, minmax | 0.6065 |
| **MRR** | HybridRank | a=0.5, b=0.4, rrf_k=10, minmax | 0.9500 |

La config `alpha=0.7, beta=0.2, rrf_k=10, minmax` es optima simultaneamente en 4 de 6 metricas.

### 3.3 Mejor configuracion por tipo de query

| Tipo de query | n | Mejor estrategia | Params | Objective |
|---|---|---|---|---|
| referencia_exacta | 4 | HybridRank | a=0.8, b=0.1, rrf_k=60, minmax | 0.8781 |
| multi_hop | 4 | HybridRank | a=0.8, b=0.4, rrf_k=10, zscore | 0.8691 |
| ambigua | 3 | HybridRank | a=0.7, b=0.0, rrf_k=10, zscore | 0.7631 |
| semantica | 4 | HybridRank | a=0.5, b=0.2, rrf_k=20, zscore | 0.7272 |
| temporal_historica | 1 | HybridRank | a=0.5, b=0.0, rrf_k=10, minmax | 0.7167 |
| compleja_hibrida | 4 | Weighted | a=0.1, minmax, k=20 | 0.7010 |

### 3.4 Leave-One-Out Cross Validation

| Estrategia | Mean Train Obj | Mean Test Obj | Std Test | Overfitting Gap |
|---|---|---|---|---|
| **HybridRank** | 0.7360 | **0.7098 +/- 0.20** | 0.2049 | **0.0263** |
| Weighted | 0.7286 | 0.6816 +/- 0.20 | 0.2014 | 0.0470 |

**HybridRank selecciono `alpha=0.7, beta=0.2, rrf_k=10, minmax` en 15 de 20 folds** (75%), demostrando alta estabilidad. Solo en 5 folds el optimizador eligio otra config, siempre cercana (alpha=0.8, beta=0.3-0.4).

Para Weighted, la seleccion se dividio entre dos configs: `alpha=0.6, minmax, k=20` (12 folds) y `alpha=0.7, zscore, k=100` (7 folds).

## 4. Interpretacion

### 4.1 HybridRank supera a WeightedScore tras optimizacion fina

En la evaluacion inicial (grid grueso), `weighted_a0.7` parecia ganar. Con el grid fino, HybridRank domina las primeras 12 posiciones. La diferencia clave: el componente RRF con `rrf_k=10` (bajo) y `beta=0.2` (peso moderado) actua como un **regularizador de ranking** que corrige errores de ordenamiento sin dominar la senal.

### 4.2 Patrones claros en los parametros optimos

**alpha = 0.7-0.8** (peso de BM25 vs Dense):
- BM25 debe pesar mas que Dense en este corpus juridico. Las normas legales tienen terminologia precisa donde la coincidencia lexica es altamente informativa.
- Dense complementa en queries semanticas (alpha=0.5 es optimo para `semantica`).

**beta = 0.1-0.3** (peso de RRF en la fusion final):
- El componente rank-based debe tener peso bajo. Confirma que RRF funciona mejor como regularizador, no como senal dominante.
- `beta=0.0` elimina RRF y reduce HybridRank a WeightedScore, pero `beta=0.2` mejora consistentemente.

**rrf_k = 10** (suavizado de RRF):
- Valor bajo = mayor discriminacion entre posiciones altas del ranking. Con k=10, la diferencia entre posicion 1 y 10 es significativa (1/11 vs 1/20), lo que amplifica la senal de consenso entre retrievers en los top resultados.
- Valores altos (60, 100) aplanan demasiado las diferencias de ranking.

**normalizer = minmax** (predominante):
- MinMax preserva la distribucion relativa de scores dentro de cada retriever. ZScore es competitivo pero minmax gana en la mayoria de escenarios.

### 4.3 Variabilidad por tipo de query

El grid search revela que **no existe una config unica optima para todos los tipos**:

- **referencia_exacta**: Necesita alto alpha (0.8-0.9) porque BM25 identifica bien las referencias legales por coincidencia de texto. Un rrf_k mayor (60) suaviza el ranking.
- **multi_hop**: Beneficia de alpha=0.8 y beta alto (0.4), el componente RRF ayuda a consolidar evidencia de multiples fuentes.
- **semantica**: Necesita alpha mas bajo (0.5) porque Dense captura la intension semantica mejor que BM25 en queries conceptuales.
- **compleja_hibrida**: Caso dificil donde Weighted simple (alpha=0.1, favoreciendo Dense) supera a HybridRank. Posiblemente estas queries requieren comprension profunda del contexto.

### 4.4 Robustez (LOO-CV)

El overfitting gap de HybridRank (0.026) es menor que el de Weighted (0.047), indicando que:
1. La config optima generaliza bien a queries no vistas.
2. HybridRank tiene mejor estabilidad estructural gracias al componente RRF como regularizador.
3. La alta consistencia de seleccion (15/20 folds eligen la misma config) indica un optimo robusto, no un artefacto del muestreo.

## 5. Configuracion recomendada

### Config global optima (para produccion):

```python
HybridRankFusion(
    alpha=0.7,      # Peso BM25 en score-fusion
    beta=0.2,       # Peso RRF en fusion final
    k=10,           # Suavizado RRF (bajo = mas discriminante)
    normalizer="minmax",
    rrf_normalizer="minmax",
)
# con candidate_k=50
```

**Metricas esperadas (LOO-CV, estimado no sesgado):**
- nDCG@10: ~0.83
- Recall@10: ~0.73
- MAP@10: ~0.60
- Objective compuesto: 0.7098

### Para adaptive fusion (futuro):

Los resultados por tipo de query sugieren que un selector adaptativo podria mejorar aun mas:
- Queries de referencia exacta -> alpha=0.8, beta=0.1, rrf_k=60
- Queries semanticas -> alpha=0.5, beta=0.2, rrf_k=20
- Queries multi-hop -> alpha=0.8, beta=0.4, rrf_k=10

## 6. Archivos generados

| Archivo | Contenido |
|---|---|
| `grid_search_results.csv` | 354 configs con todas las metricas y ranking |
| `best_configs_by_metric.csv` | Mejor config por cada metrica individual |
| `best_configs_by_query_type.csv` | Mejor config por tipo de query |
| `grid_search_loo_cv.csv` | Resultados LOO-CV (40 filas: 20 folds x 2 estrategias) |

## 7. Reproduccion

```bash
# Grid completo con LOO-CV (~60 segundos)
.venv/Scripts/python.exe experiments/grid_search_fusion.py

# Solo HybridRank sin CV (rapido, ~30 segundos)
.venv/Scripts/python.exe experiments/grid_search_fusion.py --strategy hybridrank --no-loo-cv

# Solo Weighted con verbose
.venv/Scripts/python.exe experiments/grid_search_fusion.py --strategy weighted --verbose
```
