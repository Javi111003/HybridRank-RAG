# Fase 2: Optimizacion de Parametros y Evaluacion Final

## 1. Objetivo de la Fase

La Fase 2 tiene como objetivo encontrar la configuracion optima de parametros para las estrategias de fusion **WeightedScoreFusion** y **HybridRankFusion**, y validar estadisticamente que la configuracion seleccionada generaliza bien a queries no vistas.

Esta fase responde a la pregunta: *"Dados los parametros optimizados, HybridRank es realmente mejor que una fusion ponderada simple?"*

## 2. Que incluye esta fase

### 2.1 Grid Search exhaustivo (354 configuraciones)

Se exploro un espacio completo de parametros para ambas estrategias:

| Estrategia | Parametros | Valores explorados | Total configs |
|---|---|---|---|
| **WeightedScore** | alpha (peso BM25) | 0.1 a 0.9 (paso 0.1) | 54 |
| | normalizer | minmax, zscore | |
| | candidate_k | 20, 50, 100 | |
| **HybridRank** | alpha (peso BM25 en score-fusion) | 0.5, 0.6, 0.7, 0.8, 0.9 | 300 |
| | beta (peso de RRF en fusion final) | 0.0, 0.1, 0.2, 0.3, 0.4, 0.5 | |
| | rrf_k (suavizado de RRF) | 10, 20, 40, 60, 100 | |
| | normalizer | minmax, zscore | |

### 2.2 Metrica objetivo compuesta

Para guiar la optimizacion se definio una metrica compuesta que balancea los aspectos mas relevantes para un sistema RAG juridico:

```
Objective = 0.4 * nDCG@10 + 0.3 * Recall@10 + 0.2 * MAP@10 + 0.1 * F1@10
```

**Justificacion de los pesos:**

- **nDCG@10 (40%)**: Prioriza la calidad del ordenamiento. En un contexto RAG, los primeros documentos del ranking alimentan al generador; su relevancia posicional importa mas que la mera presencia.
- **Recall@10 (30%)**: Garantiza cobertura. Para respuestas juridicas, omitir un articulo relevante puede invalidar la respuesta generada.
- **MAP@10 (20%)**: Valora la calidad sostenida del ranking completo, no solo la primera posicion.
- **F1@10 (10%)**: Penaliza el ruido excesivo (baja precision) que podria distraer al generador con documentos irrelevantes.
- **MRR se excluyo** por saturacion (>0.93 en casi todas las configuraciones).

### 2.3 Evaluacion in-sample

Se evaluan **12 estrategias** con configuraciones fijas sobre las 20 queries completas del dataset de evaluacion. Esto incluye:

- 2 baselines (BM25 solo, Dense solo)
- 5 fusiones clasicas (RRF k=60, RRF k=100, Borda, CombSUM minmax, CombSUM zscore, CombMNZ)
- 2 variantes weighted (default y optimizada)
- 2 variantes HybridRank (default y optimizada)

**Limitacion**: Los parametros "optimizados" se seleccionaron usando estas mismas 20 queries, por lo que los resultados in-sample pueden sobreestimar el rendimiento real.

### 2.4 Leave-One-Out Cross Validation (LOO-CV)

Para obtener una estimacion **no sesgada** del rendimiento de generalizacion:

```
Para cada query q_i (i = 1..20):
    1. Train: optimizar parametros sobre las otras 19 queries
    2. Test: evaluar la mejor config encontrada en q_i (held-out)
    3. Reportar: rendimiento no sesgado en q_i
```

Con solo 20 queries, un split fijo train/test seria inestable (un unico split 80/20 dejaria 4 queries de test, sensible a que queries caigan donde). LOO-CV usa cada query exactamente una vez como test, maximizando el uso de datos.

### 2.5 Comparacion pareada

Para cada una de las 20 queries, se comparan los resultados **held-out** (no in-sample) de HybridRank vs Weighted:

```
difference_i = objective_hybridrank_i - objective_weighted_i
```

Esto produce 20 diferencias pareadas que alimentan los tests estadisticos.

### 2.6 Tests estadisticos

Se aplican tres tests complementarios para determinar si la ventaja observada de HybridRank es estadisticamente significativa o puede ser atribuida al azar.

## 3. Que significa cada test estadistico

### 3.1 Sign Test (Test de signos)

**Que es**: El test no parametrico mas simple para datos pareados. Solo cuenta cuantas veces un metodo gana vs el otro, ignorando la magnitud de las diferencias.

**Hipotesis**:
- H0: P(HybridRank gana) = P(Weighted gana) = 0.5
- H1: Las probabilidades son distintas (bilateral)

**Como funciona**: Excluye empates, cuenta wins de cada lado, y usa la distribucion binomial para calcular la probabilidad de observar ese desequilibrio por azar.

**Ventaja**: No asume ninguna distribucion de los datos. Robusto ante outliers.
**Limitacion**: Desperdicia informacion al ignorar la magnitud de las diferencias.

### 3.2 Bootstrap Confidence Interval (Intervalo de confianza por bootstrap)

**Que es**: Un metodo de remuestreo que estima la distribucion del estadistico de interes (la diferencia media) sin asumir normalidad.

**Como funciona**:
1. Tomar las 20 diferencias observadas.
2. Repetir 10,000 veces: extraer una muestra con reemplazo de tamano 20, calcular la media.
3. Ordenar las 10,000 medias. El percentil 2.5 y 97.5 forman el intervalo de confianza al 95%.

**Interpretacion**:
- Si el intervalo **no incluye 0**: la diferencia es robusta (un metodo es consistentemente mejor).
- Si el intervalo **incluye 0**: no se puede descartar que la diferencia real sea nula.

**Ventaja**: No requiere supuestos distribucionales. Da una estimacion de la magnitud del efecto, no solo significancia binaria.

### 3.3 Wilcoxon Signed-Rank Test

**Que es**: Un test no parametrico para datos pareados que, a diferencia del sign test, SI considera la magnitud de las diferencias (via sus rangos).

**Hipotesis**:
- H0: La distribucion de las diferencias es simetrica alrededor de 0
- H1: Existe un desplazamiento sistematico en una direccion

**Como funciona**:
1. Calcular las diferencias d_i y eliminar las nulas.
2. Ordenar por |d_i| y asignar rangos.
3. Sumar los rangos de las diferencias positivas (W+) y negativas (W-).
4. El estadistico es min(W+, W-). Si un metodo domina consistentemente, uno de los dos sera muy pequeno.

**Ventaja**: Mas poderoso que el sign test porque usa informacion ordinal.
**Limitacion**: Requiere al menos ~6 observaciones no nulas para ser informativo.

### 3.4 Relacion entre los tres tests

| Test | Usa magnitud? | Asume distribucion? | Poder estadistico |
|------|:---:|:---:|:---:|
| Sign test | No | No | Bajo |
| Wilcoxon | Si (rangos) | Simetria | Medio |
| Bootstrap | Si (valores) | No | Alto (pero descriptivo) |

Los tres tests abordan la misma pregunta desde angulos complementarios. Si los tres coinciden, la evidencia es solida. Si divergen, tipicamente el bootstrap es el mas informativo.

## 4. Resultados obtenidos

### 4.1 Configuraciones optimas seleccionadas

| Estrategia | Parametros | Seleccionada en |
|---|---|---|
| **HybridRank optimizado** | alpha=0.7, beta=0.2, k=10, normalizer=minmax | 15/20 folds LOO-CV |
| **Weighted optimizado** | alpha=0.7, normalizer=minmax, candidate_k=20 | 12/20 folds LOO-CV |

**Interpretacion de los parametros de HybridRank:**

- **alpha=0.7**: BM25 recibe 70% del peso en la fusion de scores. El corpus juridico cubano tiene terminologia precisa donde la coincidencia lexica es altamente informativa.
- **beta=0.2**: El componente RRF participa con peso bajo (20%) en la fusion final. Funciona como *regularizador de ranking*, corrigiendo errores de ordenamiento sin dominar la senal.
- **k=10** (RRF): Valor bajo que maximiza la discriminacion entre posiciones altas. La diferencia entre posicion 1 y 10 es 1/11 vs 1/20, amplificando la senal de consenso en los top resultados.
- **normalizer=minmax**: Preserva la distribucion relativa de scores dentro de cada retriever.

### 4.2 Rendimiento in-sample (config fija, 20 queries)

| # | Estrategia | Recall | Precision | nDCG | MAP | Objective |
|---|---|---|---|---|---|---|
| 1 | weighted_optimized | 0.7336 | 0.6600 | 0.8295 | 0.6010 | **0.7356** |
| 2 | hybridrank_optimized | 0.7303 | 0.6550 | 0.8310 | 0.6028 | **0.7351** |
| 3 | weighted_a0.7 | 0.7090 | 0.6400 | 0.8155 | 0.5903 | 0.7183 |
| 4 | bm25_only | 0.7072 | 0.6300 | 0.8107 | 0.5887 | 0.7149 |
| 5 | combsum_zscore | 0.6766 | 0.6300 | 0.8148 | 0.5872 | 0.7060 |
| ... | ... | ... | ... | ... | ... | ... |
| 12 | rrf_k100 | 0.6159 | 0.5300 | 0.7188 | 0.5241 | 0.6296 |

**Observacion**: In-sample, ambas estrategias optimizadas estan practicamente empatadas (0.7356 vs 0.7351). Ambas superan ampliamente a los baselines y fusiones clasicas.

### 4.3 LOO-CV: Estimacion no sesgada

| Estrategia | Train Objective | Test Objective | Std | Overfitting Gap |
|---|---|---|---|---|
| **hybridrank_optimized** | 0.7360 | **0.7098** | 0.2049 | **0.0263** |
| weighted_optimized | 0.7286 | 0.6816 | 0.2014 | 0.0470 |

**Hallazgos clave:**

1. **HybridRank generaliza mejor** (+0.028 en test objective). La ventaja invisible in-sample se revela al usar datos held-out.
2. **Menor overfitting**: HybridRank tiene un gap train-test de 0.026 vs 0.047 de Weighted. El componente RRF actua como regularizador, estabilizando las predicciones.
3. **Alta estabilidad de seleccion**: HybridRank selecciono la misma configuracion en 15/20 folds (75%), indicando un optimo robusto, no un artefacto del muestreo.

### 4.4 Comparacion pareada (held-out)

| Metrica | Valor |
|---|---|
| HybridRank wins | **10** |
| Weighted wins | 3 |
| Empates | 7 |
| Diferencia media | +0.0282 |
| Diferencia mediana | +0.0029 |
| Rango de diferencias | [-0.1367, +0.3158] |

HybridRank gana en el 77% de las queries donde hay un ganador claro (10 de 13 no-empates).

### 4.5 Tests estadisticos

| Test | Estadistico | p-value | Interpretacion |
|---|---|---|---|
| **Sign test** | 10 wins vs 3 | **0.0923** | Marginal (p < 0.10, p > 0.05) |
| **Bootstrap CI 95%** | mean=0.0278 | CI: [-0.0096, 0.0714] | Incluye 0 marginalmente |
| **Wilcoxon** | W=25.0 | **0.1677** | No significativo |

### 4.6 Interpretacion integrada de los tests

Los tres tests cuentan una historia coherente:

1. **Hay una tendencia clara a favor de HybridRank** (10 wins vs 3, diferencia media positiva).
2. **La significancia estadistica formal no se alcanza** al nivel convencional alpha=0.05.
3. **La razon es el tamano de muestra**: Con 20 queries (y 7 empates que reducen la muestra efectiva a 13), el poder estadistico es limitado. Para detectar un efecto de esta magnitud (~0.03 en objective) con 80% de poder, se necesitarian aproximadamente 50-80 queries.
4. **El bootstrap CI incluye 0 marginalmente** (limite inferior = -0.0096), pero su centro esta firmemente en territorio positivo (+0.028).

**Conclusion practica**: La evidencia favorece a HybridRank de forma consistente pero no decisiva. Dado que:
- HybridRank gana 3.3x mas frecuentemente que Weighted
- Tiene menor overfitting (mejor generalizacion)
- Su optimo es mas estable (15/20 folds)
- No hay penalizacion de rendimiento (in-sample son equivalentes)

Se justifica adoptar HybridRank como configuracion por defecto, con la nota de que la superioridad estadistica formal requeriria un dataset de evaluacion mas grande.

## 5. Metricas desglosadas por tipo de query

| Tipo de query | n | Mejor estrategia | Insight |
|---|---|---|---|
| referencia_exacta | 4 | HybridRank (a=0.8, b=0.1, rrf_k=60) | BM25 domina; RRF suaviza |
| multi_hop | 4 | HybridRank (a=0.8, b=0.4, rrf_k=10) | RRF consolida evidencia dispersa |
| ambigua | 3 | HybridRank (a=0.7, b=0.0, rrf_k=10) | Score-fusion pura basta |
| semantica | 4 | HybridRank (a=0.5, b=0.2, rrf_k=20) | Dense necesita mas peso |
| temporal_historica | 1 | Empate | Muestra insuficiente |
| compleja_hibrida | 4 | Weighted (a=0.1, k=20) | Dense domina |

**Patron emergente**: No existe una configuracion unica optima para todos los tipos de query. Esto sugiere que una fusion adaptativa (que seleccione parametros segun el tipo de query) podria mejorar aun mas los resultados.

## 6. Configuracion congelada para produccion

```python
from src.retriever.fusion.hybrid_rank_fusion import HybridRankFusion

fusion = HybridRankFusion(
    alpha=0.7,              # 70% BM25, 30% Dense en score-fusion
    beta=0.2,               # 20% peso RRF en fusion final
    k=10,                   # Suavizado RRF bajo (alta discriminacion)
    normalizer="minmax",    # Normalizacion de scores
    rrf_normalizer="minmax",
)

# Recuperar 50 candidatos de cada retriever
candidate_k = 50
# Retornar top 10 finales al generador
top_k = 10
```

**Rendimiento esperado (LOO-CV, estimado no sesgado):**

| Metrica | Valor |
|---|---|
| nDCG@10 | 0.8087 |
| Recall@10 | 0.6965 |
| MAP@10 | 0.5847 |
| MRR | 0.9333 |
| F1@10 | 0.6040 |
| Objective compuesto | 0.7098 |

## 7. Archivos generados

| Archivo | Descripcion |
|---|---|
| `results/grid_search_results.csv` | 354 configs con todas las metricas |
| `results/grid_search_loo_cv.csv` | 40 filas LOO-CV (20 folds x 2 estrategias) |
| `results/final_phase2_insample.csv` | Ranking in-sample de 12 estrategias |
| `results/final_phase2_loocv.csv` | Resumen LOO-CV (means, stds, gaps) |
| `results/paired_hybridrank_vs_weighted.csv` | Comparacion query por query |
| `results/paired_statistical_tests.json` | Sign test, bootstrap, Wilcoxon |
| `results/phase2_final_insample_comparison.png` | Barras por objective |
| `results/phase2_loocv_comparison.png` | LOO-CV con error bars |
| `results/phase2_insample_vs_loocv.png` | Overfitting gap visual |
| `results/paired_hybridrank_vs_weighted.png` | Barras pareadas por query |
| `results/phase2_final_metrics_heatmap.png` | Heatmap de todas las metricas |

## 8. Reproduccion

```bash
# Grid search completo con LOO-CV (~60 segundos)
.venv/Scripts/python.exe experiments/grid_search_fusion.py

# Evaluacion final y cierre de fase 2 (~45 segundos)
.venv/Scripts/python.exe experiments/phase2_final_evaluation.py
```

## 9. Limitaciones y trabajo futuro

1. **Tamano de muestra**: 20 queries son insuficientes para significancia estadistica robusta. Ampliar a 50+ queries permitiria confirmar la superioridad de HybridRank con p < 0.05.
2. **Un solo corpus**: Los resultados son especificos al corpus de legislacion cubana. Otros dominios legales podrian favorecer distintos balances alpha/beta.
3. **Evaluacion intrinseca**: Las metricas de retrieval (nDCG, Recall) son proxy del rendimiento end-to-end. La Fase 3 evaluara la calidad de las respuestas generadas.
4. **Fusion adaptativa**: Los resultados por tipo de query sugieren que un selector de parametros adaptativo podria cerrar la brecha hacia el oracle (+10.5% nDCG potencial).
