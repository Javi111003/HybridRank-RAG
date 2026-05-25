"""
Grid search para optimización de parámetros de fusión HybridRank y WeightedScore.

Usa Leave-One-Out Cross Validation para selección no sesgada de parámetros
con 20 queries de evaluación. Cachea resultados de retrieval para eficiencia.

Uso:
    .venv/Scripts/python.exe experiments/grid_search_fusion.py
    .venv/Scripts/python.exe experiments/grid_search_fusion.py --strategy hybridrank
    .venv/Scripts/python.exe experiments/grid_search_fusion.py --no-loo-cv --verbose
"""

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retriever import BM25Retriever, DenseRetriever
from src.retriever.fusion.registry import get_fusion_strategy
from src.retriever.metrics import RecallAtK, PrecisionAtK, F1AtK, MRR, MAP, NDCG

EVALUATION_DATA_PATH = project_root / ".data" / "evaluation" / "norma_qrels.json"
OUTPUT_DIR = project_root / "experiments" / "results"

OBJECTIVE_WEIGHTS = {"ndcg": 0.4, "recall": 0.3, "map": 0.2, "f1": 0.1}
OBJECTIVE_K = 10

WEIGHTED_CANDIDATE_K_VALUES = [20, 50, 100]
HYBRIDRANK_CANDIDATE_K = 50

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        total = kwargs.get("total", None)
        if total is None:
            try:
                total = len(iterable)
            except TypeError:
                total = None
        desc = kwargs.get("desc", "")
        for i, item in enumerate(iterable):
            if total and i % max(1, total // 10) == 0:
                print(f"  {desc} {i}/{total} ({100*i/total:.0f}%)")
            yield item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid search para optimización de fusión HybridRank RAG"
    )
    parser.add_argument(
        "--strategy", choices=["weighted", "hybridrank", "both"], default="both",
        help="Grid de estrategia a buscar (default: both)"
    )
    parser.add_argument(
        "--no-loo-cv", action="store_true",
        help="Omitir LOO-CV (solo evaluación full grid)"
    )
    parser.add_argument(
        "--k-eval", type=int, default=10,
        help="Valor de k para métricas de evaluación (default: 10)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directorio de salida (default: experiments/results/)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Progreso detallado por configuración"
    )
    return parser.parse_args()


def load_evaluation_data() -> List[Dict[str, Any]]:
    with open(EVALUATION_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    queries = [q for q in data if q.get("relevant_docs")]
    print(f"Cargadas {len(queries)} queries con relevancia")
    return queries


def build_weighted_grid() -> List[Dict[str, Any]]:
    grid = []
    for alpha in [round(x * 0.1, 1) for x in range(1, 10)]:
        for normalizer in ["minmax", "zscore"]:
            for candidate_k in WEIGHTED_CANDIDATE_K_VALUES:
                grid.append({
                    "alpha": alpha,
                    "normalizer": normalizer,
                    "candidate_k": candidate_k,
                })
    return grid


def build_hybridrank_grid() -> List[Dict[str, Any]]:
    grid = []
    for alpha in [0.5, 0.6, 0.7, 0.8, 0.9]:
        for beta in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            for rrf_k in [10, 20, 40, 60, 100]:
                for normalizer in ["minmax", "zscore"]:
                    grid.append({
                        "alpha": alpha,
                        "beta": beta,
                        "rrf_k": rrf_k,
                        "normalizer": normalizer,
                        "candidate_k": HYBRIDRANK_CANDIDATE_K,
                    })
    return grid


def cache_retrieval_results(
    bm25: BM25Retriever,
    dense: DenseRetriever,
    queries: List[Dict[str, Any]],
    candidate_k_values: List[int],
) -> Dict[int, Dict[str, Dict[str, List[Tuple[str, float]]]]]:
    """
    Pre-recupera resultados de BM25 y Dense para cada candidate_k.

    Returns:
        {candidate_k: {query_id: {"bm25": [(doc_id, score), ...], "dense": [...]}}}
    """
    cache: Dict[int, Dict[str, Dict[str, List[Tuple[str, float]]]]] = {}

    for ck in candidate_k_values:
        print(f"  Cacheando resultados para candidate_k={ck}...")
        cache[ck] = {}
        for q in queries:
            qid = q["query_id"]
            query_text = q["query"]
            bm25_results = bm25.retrieve(query_text, top_k=ck)
            dense_results = dense.retrieve(query_text, top_k=ck)
            cache[ck][qid] = {"bm25": bm25_results, "dense": dense_results}

    return cache


def build_strategy_from_config(strategy_type: str, config: Dict[str, Any]):
    if strategy_type == "weighted":
        return get_fusion_strategy(
            "weighted",
            alpha=config["alpha"],
            normalizer=config["normalizer"],
        )
    else:
        return get_fusion_strategy(
            "hybridrank",
            alpha=config["alpha"],
            beta=config["beta"],
            k=config["rrf_k"],
            normalizer=config["normalizer"],
            rrf_normalizer=config["normalizer"],
        )


def compute_metrics_for_query(
    fused_results: List[Tuple[str, float]],
    relevant_docs: List[str],
    metrics: List[Any],
    k: int,
) -> Dict[str, float]:
    result = {}
    for metric in metrics:
        score = metric.compute(fused_results, relevant_docs, k=k)["score"]
        col_name = metric.name.lower().replace("@k", "")
        result[col_name] = score
    return result


def compute_objective(metrics_dict: Dict[str, float]) -> float:
    return (
        OBJECTIVE_WEIGHTS["ndcg"] * metrics_dict.get("ndcg", 0.0)
        + OBJECTIVE_WEIGHTS["recall"] * metrics_dict.get("recall", 0.0)
        + OBJECTIVE_WEIGHTS["map"] * metrics_dict.get("map", 0.0)
        + OBJECTIVE_WEIGHTS["f1"] * metrics_dict.get("f1", 0.0)
    )


def evaluate_config_on_queries(
    strategy_type: str,
    config: Dict[str, Any],
    cache: Dict[int, Dict[str, Dict[str, List[Tuple[str, float]]]]],
    queries: List[Dict[str, Any]],
    metrics: List[Any],
    k: int,
) -> Tuple[Dict[str, float], float]:
    """
    Evalúa una config sobre un conjunto de queries.

    Returns:
        (mean_metrics_dict, mean_objective)
    """
    candidate_k = config["candidate_k"]
    strategy = build_strategy_from_config(strategy_type, config)
    top_k = max(k, 20)

    all_metrics: List[Dict[str, float]] = []
    for q in queries:
        qid = q["query_id"]
        results_by_retriever = cache[candidate_k][qid]
        fused = strategy.fuse(results_by_retriever, top_k=top_k)
        m = compute_metrics_for_query(fused, q["relevant_docs"], metrics, k)
        all_metrics.append(m)

    mean_metrics = {}
    for key in all_metrics[0]:
        mean_metrics[key] = np.mean([m[key] for m in all_metrics])

    mean_obj = np.mean([compute_objective(m) for m in all_metrics])
    return mean_metrics, mean_obj


def run_full_grid(
    strategy_type: str,
    grid: List[Dict[str, Any]],
    cache: Dict[int, Dict[str, Dict[str, List[Tuple[str, float]]]]],
    queries: List[Dict[str, Any]],
    metrics: List[Any],
    k: int,
    verbose: bool = False,
) -> pd.DataFrame:
    """Evalúa todas las configs del grid sobre todas las queries."""
    rows = []

    iterator = tqdm(grid, desc=f"Grid {strategy_type}", total=len(grid))
    for config in iterator:
        mean_metrics, mean_obj = evaluate_config_on_queries(
            strategy_type, config, cache, queries, metrics, k
        )

        row = {"strategy_type": strategy_type, **config}
        for key, val in mean_metrics.items():
            row[f"mean_{key}_{k}"] = round(val, 4)
        row["objective"] = round(mean_obj, 4)
        rows.append(row)

        if verbose:
            print(f"    {config} -> obj={mean_obj:.4f}")

    df = pd.DataFrame(rows)
    df["rank"] = df["objective"].rank(ascending=False, method="min").astype(int)
    df = df.sort_values("objective", ascending=False).reset_index(drop=True)
    return df


def run_loo_cv(
    strategy_type: str,
    grid: List[Dict[str, Any]],
    cache: Dict[int, Dict[str, Dict[str, List[Tuple[str, float]]]]],
    queries: List[Dict[str, Any]],
    metrics: List[Any],
    k: int,
    verbose: bool = False,
) -> pd.DataFrame:
    """Leave-One-Out Cross Validation."""
    loo_results = []
    n_queries = len(queries)

    for i in tqdm(range(n_queries), desc=f"LOO-CV {strategy_type}", total=n_queries):
        held_out = queries[i]
        train_queries = [q for j, q in enumerate(queries) if j != i]

        best_obj = -1.0
        best_config = None

        for config in grid:
            _, mean_obj = evaluate_config_on_queries(
                strategy_type, config, cache, train_queries, metrics, k
            )
            if mean_obj > best_obj:
                best_obj = mean_obj
                best_config = config

        candidate_k = best_config["candidate_k"]
        strategy = build_strategy_from_config(strategy_type, best_config)
        top_k = max(k, 20)
        results_by_retriever = cache[candidate_k][held_out["query_id"]]
        fused = strategy.fuse(results_by_retriever, top_k=top_k)
        test_metrics = compute_metrics_for_query(
            fused, held_out["relevant_docs"], metrics, k
        )
        test_obj = compute_objective(test_metrics)

        row = {
            "held_out_query_id": held_out["query_id"],
            "query_type": held_out.get("query_type", "unknown"),
            "strategy_type": strategy_type,
            "best_alpha": best_config["alpha"],
            "best_normalizer": best_config["normalizer"],
            "best_candidate_k": best_config["candidate_k"],
            "train_objective": round(best_obj, 4),
        }
        if strategy_type == "hybridrank":
            row["best_beta"] = best_config["beta"]
            row["best_rrf_k"] = best_config["rrf_k"]

        for key, val in test_metrics.items():
            row[f"test_{key}"] = round(val, 4)
        row["test_objective"] = round(test_obj, 4)
        loo_results.append(row)

        if verbose:
            print(
                f"    Fold {i+1}/{n_queries}: query={held_out['query_id']} "
                f"best_train_obj={best_obj:.4f} test_obj={test_obj:.4f}"
            )

    return pd.DataFrame(loo_results)


def find_best_by_metric(full_results_df: pd.DataFrame, k: int) -> pd.DataFrame:
    metric_cols = [c for c in full_results_df.columns if c.startswith("mean_") and c != "objective"]
    rows = []

    for col in metric_cols:
        best_idx = full_results_df[col].idxmax()
        best_row = full_results_df.loc[best_idx]
        metric_name = col.replace(f"mean_", "").replace(f"_{k}", "")
        rows.append({
            "metric": metric_name,
            "strategy_type": best_row["strategy_type"],
            "alpha": best_row["alpha"],
            "beta": best_row.get("beta", None),
            "rrf_k": best_row.get("rrf_k", None),
            "normalizer": best_row["normalizer"],
            "candidate_k": best_row["candidate_k"],
            "best_score": round(best_row[col], 4),
            "objective": best_row["objective"],
        })

    return pd.DataFrame(rows)


def find_best_by_query_type(
    strategy_type: str,
    grid: List[Dict[str, Any]],
    cache: Dict[int, Dict[str, Dict[str, List[Tuple[str, float]]]]],
    queries: List[Dict[str, Any]],
    metrics: List[Any],
    k: int,
) -> pd.DataFrame:
    """Encuentra la mejor config por tipo de query."""
    query_types = set(q.get("query_type", "unknown") for q in queries)
    rows = []

    for qt in sorted(query_types):
        qt_queries = [q for q in queries if q.get("query_type", "unknown") == qt]
        if not qt_queries:
            continue

        best_obj = -1.0
        best_config = None
        best_metrics = None

        for config in grid:
            mean_metrics, mean_obj = evaluate_config_on_queries(
                strategy_type, config, cache, qt_queries, metrics, k
            )
            if mean_obj > best_obj:
                best_obj = mean_obj
                best_config = config
                best_metrics = mean_metrics

        row = {
            "query_type": qt,
            "n_queries": len(qt_queries),
            "strategy_type": strategy_type,
            "alpha": best_config["alpha"],
            "normalizer": best_config["normalizer"],
            "candidate_k": best_config["candidate_k"],
        }
        if strategy_type == "hybridrank":
            row["beta"] = best_config["beta"]
            row["rrf_k"] = best_config["rrf_k"]

        row[f"mean_ndcg_{k}"] = round(best_metrics.get("ndcg", 0), 4)
        row[f"mean_recall_{k}"] = round(best_metrics.get("recall", 0), 4)
        row["objective"] = round(best_obj, 4)
        rows.append(row)

    return pd.DataFrame(rows)


def print_summary(
    full_df: pd.DataFrame,
    loo_df: Optional[pd.DataFrame],
    by_metric_df: pd.DataFrame,
    by_type_df: pd.DataFrame,
):
    print("\n" + "=" * 70)
    print("  RESULTADOS DEL GRID SEARCH")
    print("=" * 70)

    print("\n--- TOP 10 CONFIGURACIONES (por objective) ---")
    top_cols = ["strategy_type", "alpha", "objective"]
    if "beta" in full_df.columns:
        top_cols = ["strategy_type", "alpha", "beta", "rrf_k", "normalizer", "candidate_k", "objective"]
    else:
        top_cols = ["strategy_type", "alpha", "normalizer", "candidate_k", "objective"]
    available_cols = [c for c in top_cols if c in full_df.columns]
    print(full_df.head(10)[available_cols].to_string(index=False))

    print("\n--- MEJOR CONFIG POR MÉTRICA ---")
    print(by_metric_df.to_string(index=False))

    print("\n--- MEJOR CONFIG POR TIPO DE QUERY ---")
    print(by_type_df.to_string(index=False))

    if loo_df is not None:
        print("\n--- LOO-CV RESUMEN ---")
        mean_test_obj = loo_df["test_objective"].mean()
        std_test_obj = loo_df["test_objective"].std()
        mean_train_obj = loo_df["train_objective"].mean()
        print(f"  Mean train objective: {mean_train_obj:.4f}")
        print(f"  Mean test objective:  {mean_test_obj:.4f} ± {std_test_obj:.4f}")
        print(f"  Overfitting gap:      {mean_train_obj - mean_test_obj:.4f}")

        if "best_beta" in loo_df.columns:
            print("\n  Configs seleccionadas por fold:")
            param_cols = ["held_out_query_id", "best_alpha", "best_beta", "best_rrf_k",
                          "best_normalizer", "test_objective"]
        else:
            param_cols = ["held_out_query_id", "best_alpha", "best_normalizer",
                          "best_candidate_k", "test_objective"]
        available = [c for c in param_cols if c in loo_df.columns]
        print(loo_df[available].to_string(index=False))


def main():
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(exist_ok=True, parents=True)
    k = args.k_eval

    print("=" * 70)
    print("  GRID SEARCH - OPTIMIZACIÓN DE FUSIÓN")
    print("=" * 70)
    print(f"  Estrategia: {args.strategy}")
    print(f"  LOO-CV: {'No' if args.no_loo_cv else 'Sí'}")
    print(f"  k evaluación: {k}")
    print(f"  Output: {output_dir}")
    print()

    queries = load_evaluation_data()
    metrics = [RecallAtK(), PrecisionAtK(), F1AtK(), MRR(), MAP(), NDCG()]

    # Determinar candidate_k values necesarios
    all_candidate_k = set()
    grids = {}
    if args.strategy in ("weighted", "both"):
        grids["weighted"] = build_weighted_grid()
        all_candidate_k.update(WEIGHTED_CANDIDATE_K_VALUES)
        print(f"  Grid WeightedScore: {len(grids['weighted'])} configs")
    if args.strategy in ("hybridrank", "both"):
        grids["hybridrank"] = build_hybridrank_grid()
        all_candidate_k.add(HYBRIDRANK_CANDIDATE_K)
        print(f"  Grid HybridRank: {len(grids['hybridrank'])} configs")

    # Inicializar retrievers y cachear
    print("\nInicializando recuperadores...")
    bm25 = BM25Retriever(index_dir=str(project_root / ".data" / "bm25_norma_index"))
    dense = DenseRetriever(
        chroma_dir=str(project_root / ".data" / "chroma_normas"),
        collection_name="hybridrank_normas",
    )
    print("  BM25Retriever: OK")
    print("  DenseRetriever: OK")

    print("\nCacheando resultados de retrieval...")
    t0 = time.time()
    cache = cache_retrieval_results(
        bm25, dense, queries, sorted(all_candidate_k)
    )
    print(f"  Cache completado en {time.time() - t0:.1f}s")

    # Ejecutar grid search por estrategia
    all_full_dfs = []
    all_loo_dfs = []
    all_by_metric_dfs = []
    all_by_type_dfs = []

    for strategy_type, grid in grids.items():
        print(f"\n{'-' * 70}")
        print(f"  Evaluando grid: {strategy_type} ({len(grid)} configs)")
        print(f"{'-' * 70}")

        t0 = time.time()
        full_df = run_full_grid(
            strategy_type, grid, cache, queries, metrics, k, args.verbose
        )
        print(f"  Grid completado en {time.time() - t0:.1f}s")
        all_full_dfs.append(full_df)

        by_metric_df = find_best_by_metric(full_df, k)
        all_by_metric_dfs.append(by_metric_df)

        print("  Calculando mejor por tipo de query...")
        by_type_df = find_best_by_query_type(
            strategy_type, grid, cache, queries, metrics, k
        )
        all_by_type_dfs.append(by_type_df)

        loo_df = None
        if not args.no_loo_cv:
            print(f"\n  Ejecutando LOO-CV ({len(queries)} folds x {len(grid)} configs)...")
            t0 = time.time()
            loo_df = run_loo_cv(
                strategy_type, grid, cache, queries, metrics, k, args.verbose
            )
            print(f"  LOO-CV completado en {time.time() - t0:.1f}s")
            all_loo_dfs.append(loo_df)

        print_summary(full_df, loo_df, by_metric_df, by_type_df)

    # Combinar y guardar resultados
    combined_full = pd.concat(all_full_dfs, ignore_index=True)
    combined_full = combined_full.sort_values("objective", ascending=False).reset_index(drop=True)
    combined_full["rank"] = range(1, len(combined_full) + 1)

    full_path = output_dir / "grid_search_results.csv"
    combined_full.to_csv(full_path, index=False, encoding="utf-8")
    print(f"\nGuardado: {full_path} ({len(combined_full)} filas)")

    combined_by_metric = pd.concat(all_by_metric_dfs, ignore_index=True)
    by_metric_path = output_dir / "best_configs_by_metric.csv"
    combined_by_metric.to_csv(by_metric_path, index=False, encoding="utf-8")
    print(f"Guardado: {by_metric_path}")

    combined_by_type = pd.concat(all_by_type_dfs, ignore_index=True)
    by_type_path = output_dir / "best_configs_by_query_type.csv"
    combined_by_type.to_csv(by_type_path, index=False, encoding="utf-8")
    print(f"Guardado: {by_type_path}")

    if all_loo_dfs:
        combined_loo = pd.concat(all_loo_dfs, ignore_index=True)
        loo_path = output_dir / "grid_search_loo_cv.csv"
        combined_loo.to_csv(loo_path, index=False, encoding="utf-8")
        print(f"Guardado: {loo_path}")

    print("\n" + "=" * 70)
    print("  GRID SEARCH COMPLETADO")
    print("=" * 70)


if __name__ == "__main__":
    main()
