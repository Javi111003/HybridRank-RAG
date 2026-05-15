"""
Script de evaluación comparativa de estrategias de fusión.

Evalúa todas las estrategias de fusión implementadas sobre el dataset
de evaluación (qrels.json), comparándolas con los baselines (BM25 solo, Dense solo).

Uso:
    .venv/Scripts/python.exe experiments/evaluate_fusion_strategies.py

Salida:
    experiments/results/fusion_metrics.csv
    experiments/results/fusion_summary.csv
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Agregar raíz del proyecto al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.retriever import BM25Retriever, DenseRetriever
from src.retriever.hybrid_retriever import HybridRetriever
from src.retriever.fusion.registry import get_fusion_strategy
from src.retriever.metrics import RecallAtK, PrecisionAtK, F1AtK, MRR, MAP, NDCG


EVALUATION_DATA_PATH = project_root / ".data" / "evaluation" / "norma_qrels.json"
OUTPUT_DIR = project_root / "experiments" / "results"
K_VALUES = [5, 10, 20]


def load_evaluation_data() -> List[Dict[str, Any]]:
    """Carga dataset de evaluación desde qrels.json."""
    with open(EVALUATION_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Cargadas {len(data)} queries de evaluación")
    return data


def define_strategies() -> Dict[str, Dict[str, Any]]:
    """
    Define todas las configuraciones de estrategias a evaluar.

    Returns:
        Dict nombre_config -> {"type": str, "params": dict}
        type="baseline" para recuperadores individuales.
        type=nombre_estrategia para estrategias de fusión.
    """
    return {
        # Baselines
        "bm25_only": {"type": "baseline", "retriever": "bm25"},
        "dense_only": {"type": "baseline", "retriever": "dense"},
        # RRF
        "rrf_k60": {"type": "rrf", "params": {"k": 60}},
        "rrf_k100": {"type": "rrf", "params": {"k": 100}},
        # Borda
        "borda": {"type": "borda", "params": {}},
        # CombSUM
        "combsum_minmax": {"type": "combsum", "params": {"normalizer": "minmax"}},
        "combsum_zscore": {"type": "combsum", "params": {"normalizer": "zscore"}},
        # CombMNZ
        "combmnz_minmax": {"type": "combmnz", "params": {"normalizer": "minmax"}},
        # Weighted Score
        "weighted_a0.3": {"type": "weighted", "params": {"alpha": 0.3}},
        "weighted_a0.5": {"type": "weighted", "params": {"alpha": 0.5}},
        "weighted_a0.7": {"type": "weighted", "params": {"alpha": 0.7}},
        # HybridRank (propuesta)
        "hybridrank_a0.5_b0.5": {"type": "hybridrank", "params": {"alpha": 0.5, "beta": 0.5}},
        "hybridrank_a0.7_b0.5": {"type": "hybridrank", "params": {"alpha": 0.7, "beta": 0.5}},
        "hybridrank_a0.5_b0.7": {"type": "hybridrank", "params": {"alpha": 0.5, "beta": 0.7}},
        "hybridrank_a0.7_b0.3": {"type": "hybridrank", "params": {"alpha": 0.7, "beta": 0.3}},
        "hybridrank_a0.3_b0.7": {"type": "hybridrank", "params": {"alpha": 0.3, "beta": 0.7}},
    }


def build_retriever(
    config: Dict[str, Any],
    bm25: BM25Retriever,
    dense: DenseRetriever,
    candidate_k: int = 50,
) -> Tuple[Any, str]:
    """
    Construye un recuperador según la configuración.

    Returns:
        Tupla (retriever, nombre)
    """
    if config["type"] == "baseline":
        if config["retriever"] == "bm25":
            return bm25, "bm25_only"
        else:
            return dense, "dense_only"

    fusion_strategy = get_fusion_strategy(config["type"], **config.get("params", {}))
    hybrid = HybridRetriever(
        retrievers={"bm25": bm25, "dense": dense},
        fusion_strategy=fusion_strategy,
        candidate_k=candidate_k,
    )
    return hybrid, hybrid.name


def evaluate_single_query(
    retriever: Any,
    query_text: str,
    relevant_docs: List[str],
    metrics: List[Any],
    k_values: List[int],
) -> List[Dict[str, Any]]:
    """
    Evalúa un recuperador sobre una query con todas las métricas y k_values.

    Returns:
        Lista de dicts con resultados por cada k.
    """
    max_k = max(k_values)
    retrieved = retriever.retrieve(query_text, top_k=max_k)

    rows = []
    for k in k_values:
        row: Dict[str, Any] = {"top_k": k}
        for metric in metrics:
            result = metric.compute(retrieved, relevant_docs, k=k)
            metric_name = result["metric_name"].lower().replace("@k", f"@{k}")
            row[metric_name] = result["score"]
        rows.append(row)

    return rows


def run_evaluation():
    """Ejecuta la evaluación completa."""
    print("=" * 60)
    print("  EVALUACIÓN DE ESTRATEGIAS DE FUSIÓN - HybridRank RAG")
    print("=" * 60)

    # Cargar datos
    eval_data = load_evaluation_data()

    # Filtrar queries con documentos relevantes
    queries_with_relevance = [q for q in eval_data if q.get("relevant_docs")]
    print(f"Queries con relevancia: {len(queries_with_relevance)} / {len(eval_data)}")

    if not queries_with_relevance:
        print("\n[ADVERTENCIA] No hay queries con documentos relevantes anotados.")
        print("La evaluación se ejecutará pero todas las métricas serán 0.")
        print("Para obtener resultados útiles, anota relevant_docs en qrels.json.\n")
        queries_with_relevance = eval_data

    # Inicializar recuperadores base
    print("\nInicializando recuperadores...")
    bm25 = BM25Retriever()
    dense = DenseRetriever()
    print("  - BM25Retriever: OK")
    print("  - DenseRetriever: OK")

    # Métricas
    metrics = [RecallAtK(), PrecisionAtK(), F1AtK(), MRR(), MAP(), NDCG()]
    print(f"  - Métricas: {[m.name for m in metrics]}")

    # Estrategias
    strategies = define_strategies()
    print(f"\nEvaluando {len(strategies)} configuraciones...")
    print("-" * 60)

    all_results: List[Dict[str, Any]] = []

    for strategy_name, config in strategies.items():
        print(f"  [{strategy_name}]", end=" ... ")
        retriever, retriever_full_name = build_retriever(config, bm25, dense)

        for query_data in queries_with_relevance:
            query_id = query_data["query_id"]
            query_text = query_data["query"]
            query_type = query_data.get("query_type", "unknown")
            relevant_docs = query_data.get("relevant_docs", [])

            rows = evaluate_single_query(
                retriever, query_text, relevant_docs, metrics, K_VALUES
            )

            for row in rows:
                row["strategy"] = strategy_name
                row["query_id"] = query_id
                row["query_type"] = query_type
                row["retriever_name"] = retriever_full_name
                all_results.append(row)

        print("OK")

    print("-" * 60)

    # Crear DataFrame
    df = pd.DataFrame(all_results)

    # Guardar resultados detallados
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    detail_path = OUTPUT_DIR / "fusion_metrics.csv"
    df.to_csv(detail_path, index=False, encoding="utf-8")
    print(f"\nResultados detallados: {detail_path}")
    print(f"  → {len(df)} filas")

    # Crear resumen agregado por estrategia
    metric_cols = [c for c in df.columns if any(
        m in c for m in ["recall", "precision", "f1", "mrr", "map", "ndcg"]
    )]

    if metric_cols:
        summary = df.groupby("strategy")[metric_cols].mean().round(4)
        summary = summary.sort_values(metric_cols[0], ascending=False)
        summary_path = OUTPUT_DIR / "fusion_summary.csv"
        summary.to_csv(summary_path, encoding="utf-8")
        print(f"Resumen por estrategia: {summary_path}")

        print("\n" + "=" * 60)
        print("  RESUMEN (promedios por estrategia)")
        print("=" * 60)
        print(summary.to_string())

    print("\n\nEvaluación completada.")


if __name__ == "__main__":
    run_evaluation()
