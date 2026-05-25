"""
Cierre formal de Fase 2: Evaluacion final de optimizacion de parametros.

Produce:
- Tabla in-sample con configs fijas sobre 20 queries
- Resumen LOO-CV desde grid_search_loo_cv.csv
- Comparacion pareada HybridRank vs Weighted (held-out)
- Tests estadisticos (sign test, bootstrap, wilcoxon)
- 5 visualizaciones

Uso:
    .venv/Scripts/python.exe experiments/phase2_final_evaluation.py
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retriever import BM25Retriever, DenseRetriever
from src.retriever.fusion.registry import get_fusion_strategy
from src.retriever.metrics import RecallAtK, PrecisionAtK, F1AtK, MRR, MAP, NDCG

EVALUATION_DATA_PATH = project_root / ".data" / "evaluation" / "norma_qrels.json"
OUTPUT_DIR = project_root / "experiments" / "results"
LOOCV_PATH = OUTPUT_DIR / "grid_search_loo_cv.csv"

OBJECTIVE_WEIGHTS = {"ndcg": 0.4, "recall": 0.3, "map": 0.2, "f1": 0.1}
K_EVAL = 10

STRATEGIES_FINAL = {
    "bm25_only": {"type": "baseline", "retriever": "bm25", "family": "baseline", "candidate_k": 50},
    "dense_only": {"type": "baseline", "retriever": "dense", "family": "baseline", "candidate_k": 50},
    "rrf_k60": {"type": "rrf", "params": {"k": 60}, "family": "classic", "candidate_k": 50},
    "rrf_k100": {"type": "rrf", "params": {"k": 100}, "family": "classic", "candidate_k": 50},
    "borda": {"type": "borda", "params": {}, "family": "classic", "candidate_k": 50},
    "combsum_minmax": {"type": "combsum", "params": {"normalizer": "minmax"}, "family": "classic", "candidate_k": 50},
    "combsum_zscore": {"type": "combsum", "params": {"normalizer": "zscore"}, "family": "classic", "candidate_k": 50},
    "combmnz_minmax": {"type": "combmnz", "params": {"normalizer": "minmax"}, "family": "classic", "candidate_k": 50},
    "weighted_a0.7": {"type": "weighted", "params": {"alpha": 0.7}, "family": "weighted", "candidate_k": 50},
    "weighted_optimized": {"type": "weighted", "params": {"alpha": 0.7, "normalizer": "minmax"}, "family": "weighted", "candidate_k": 20},
    "hybridrank_a0.7_b0.3": {"type": "hybridrank", "params": {"alpha": 0.7, "beta": 0.3}, "family": "hybridrank", "candidate_k": 50},
    "hybridrank_optimized": {"type": "hybridrank", "params": {"alpha": 0.7, "beta": 0.2, "k": 10, "normalizer": "minmax", "rrf_normalizer": "minmax"}, "family": "hybridrank", "candidate_k": 50},
}


def load_evaluation_data() -> List[Dict[str, Any]]:
    with open(EVALUATION_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [q for q in data if q.get("relevant_docs")]


def compute_objective(metrics: Dict[str, float]) -> float:
    return (
        OBJECTIVE_WEIGHTS["ndcg"] * metrics.get("ndcg", 0.0)
        + OBJECTIVE_WEIGHTS["recall"] * metrics.get("recall", 0.0)
        + OBJECTIVE_WEIGHTS["map"] * metrics.get("map", 0.0)
        + OBJECTIVE_WEIGHTS["f1"] * metrics.get("f1", 0.0)
    )


def cache_retrieval_results(
    bm25: BM25Retriever,
    dense: DenseRetriever,
    queries: List[Dict[str, Any]],
    candidate_k_values: List[int],
) -> Dict[int, Dict[str, Dict[str, List[Tuple[str, float]]]]]:
    cache = {}
    for ck in candidate_k_values:
        print(f"  Cacheando candidate_k={ck}...")
        cache[ck] = {}
        for q in queries:
            qid = q["query_id"]
            cache[ck][qid] = {
                "bm25": bm25.retrieve(q["query"], top_k=ck),
                "dense": dense.retrieve(q["query"], top_k=ck),
            }
    return cache


def config_to_string(name: str, config: Dict[str, Any]) -> str:
    if config["type"] == "baseline":
        return name
    params = config.get("params", {})
    parts = [f"{k}={v}" for k, v in sorted(params.items())]
    return f"{config['type']}({', '.join(parts)})"


def run_insample_evaluation(
    queries: List[Dict[str, Any]],
    cache: Dict[int, Dict[str, Dict[str, List[Tuple[str, float]]]]],
    metrics_list: List[Any],
) -> pd.DataFrame:
    """Evalua todas las estrategias finales in-sample (config fija, 20 queries)."""
    rows = []

    for strategy_name, config in STRATEGIES_FINAL.items():
        candidate_k = config["candidate_k"]
        top_k = max(K_EVAL, 20)

        if config["type"] == "baseline":
            retriever_key = config["retriever"]
        else:
            strategy = get_fusion_strategy(config["type"], **config.get("params", {}))

        all_metrics = []
        for q in queries:
            qid = q["query_id"]
            relevant = q["relevant_docs"]

            if config["type"] == "baseline":
                retriever_key = config["retriever"]
                fused = cache[candidate_k][qid][retriever_key][:top_k]
            else:
                results_by_retriever = cache[candidate_k][qid]
                fused = strategy.fuse(results_by_retriever, top_k=top_k)

            m = {}
            for metric in metrics_list:
                score = metric.compute(fused, relevant, k=K_EVAL)["score"]
                col = metric.name.lower().replace("@k", "")
                m[col] = score
            all_metrics.append(m)

        mean_m = {k: np.mean([x[k] for x in all_metrics]) for k in all_metrics[0]}
        obj = np.mean([compute_objective(x) for x in all_metrics])

        rows.append({
            "strategy": strategy_name,
            "strategy_family": config["family"],
            "config": config_to_string(strategy_name, config),
            "candidate_k": candidate_k,
            "recall": round(mean_m["recall"], 4),
            "precision": round(mean_m["precision"], 4),
            "f1": round(mean_m["f1"], 4),
            "mrr": round(mean_m["mrr"], 4),
            "map": round(mean_m["map"], 4),
            "ndcg": round(mean_m["ndcg"], 4),
            "objective": round(obj, 4),
        })
        print(f"  {strategy_name}: objective={obj:.4f}")

    df = pd.DataFrame(rows).sort_values("objective", ascending=False).reset_index(drop=True)
    return df


def build_loocv_summary() -> pd.DataFrame:
    """Lee grid_search_loo_cv.csv y produce resumen por estrategia."""
    loo = pd.read_csv(LOOCV_PATH)

    rows = []
    for st in ["weighted", "hybridrank"]:
        subset = loo[loo["strategy_type"] == st]
        strategy_name = f"{st}_optimized"

        rows.append({
            "strategy": strategy_name,
            "mean_train_objective": round(subset["train_objective"].mean(), 4),
            "mean_test_objective": round(subset["test_objective"].mean(), 4),
            "std_test_objective": round(subset["test_objective"].std(), 4),
            "overfitting_gap": round(
                subset["train_objective"].mean() - subset["test_objective"].mean(), 4
            ),
            "mean_test_recall": round(subset["test_recall"].mean(), 4),
            "mean_test_precision": round(subset["test_precision"].mean(), 4),
            "mean_test_f1": round(subset["test_f1"].mean(), 4),
            "mean_test_mrr": round(subset["test_mrr"].mean(), 4),
            "mean_test_map": round(subset["test_map"].mean(), 4),
            "mean_test_ndcg": round(subset["test_ndcg"].mean(), 4),
        })

    return pd.DataFrame(rows)


def run_paired_comparison() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Comparacion pareada query por query usando datos held-out."""
    loo = pd.read_csv(LOOCV_PATH)

    weighted = loo[loo["strategy_type"] == "weighted"].set_index("held_out_query_id")
    hybridrank = loo[loo["strategy_type"] == "hybridrank"].set_index("held_out_query_id")

    rows = []
    for qid in weighted.index:
        obj_w = weighted.loc[qid, "test_objective"]
        obj_h = hybridrank.loc[qid, "test_objective"]
        diff = obj_h - obj_w

        if abs(diff) <= 1e-6:
            winner = "tie"
        elif diff > 0:
            winner = "hybridrank"
        else:
            winner = "weighted"

        rows.append({
            "query_id": qid,
            "query_type": weighted.loc[qid, "query_type"],
            "objective_weighted": round(obj_w, 4),
            "objective_hybridrank": round(obj_h, 4),
            "difference": round(diff, 4),
            "winner": winner,
        })

    paired_df = pd.DataFrame(rows)

    differences = paired_df["difference"].values
    n_hr_wins = int((paired_df["winner"] == "hybridrank").sum())
    n_w_wins = int((paired_df["winner"] == "weighted").sum())
    n_ties = int((paired_df["winner"] == "tie").sum())

    summary = {
        "n_queries": len(paired_df),
        "hybridrank_wins": n_hr_wins,
        "weighted_wins": n_w_wins,
        "ties": n_ties,
        "mean_difference": round(float(np.mean(differences)), 4),
        "median_difference": round(float(np.median(differences)), 4),
        "std_difference": round(float(np.std(differences)), 4),
        "min_difference": round(float(np.min(differences)), 4),
        "max_difference": round(float(np.max(differences)), 4),
    }

    return paired_df, summary


def run_statistical_tests(paired_df: pd.DataFrame) -> Dict[str, Any]:
    """Sign test, bootstrap, y Wilcoxon sobre las diferencias held-out."""
    differences = paired_df["difference"].values
    non_zero = differences[np.abs(differences) > 1e-6]

    n_hr_wins = int(np.sum(non_zero > 0))
    n_w_wins = int(np.sum(non_zero < 0))
    n_ties = int(len(differences) - len(non_zero))

    # Sign test
    sign_p = None
    try:
        from scipy.stats import binomtest
        result = binomtest(n_hr_wins, n_hr_wins + n_w_wins, 0.5, alternative="two-sided")
        sign_p = round(float(result.pvalue), 4)
    except (ImportError, Exception):
        pass

    # Bootstrap
    rng = np.random.default_rng(42)
    n_bootstrap = 10000
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(differences, size=len(differences), replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.array(boot_means)
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    # Wilcoxon
    wilcoxon_stat = None
    wilcoxon_p = None
    try:
        from scipy.stats import wilcoxon as wilcoxon_test
        if len(non_zero) >= 6:
            stat, p = wilcoxon_test(non_zero)
            wilcoxon_stat = round(float(stat), 4)
            wilcoxon_p = round(float(p), 4)
    except (ImportError, Exception):
        pass

    return {
        "n_queries": int(len(differences)),
        "sign_test": {
            "hybridrank_wins": n_hr_wins,
            "weighted_wins": n_w_wins,
            "ties": n_ties,
            "p_value": sign_p,
        },
        "bootstrap": {
            "mean": round(float(np.mean(boot_means)), 4),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "n_bootstrap": n_bootstrap,
        },
        "wilcoxon": {
            "statistic": wilcoxon_stat,
            "p_value": wilcoxon_p,
        },
    }


def generate_visualizations(
    insample_df: pd.DataFrame,
    loocv_df: pd.DataFrame,
    paired_df: pd.DataFrame,
):
    """Genera las 5 figuras de cierre de fase 2."""
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 150

    family_colors = {
        "baseline": "#888888",
        "classic": "#4A90D9",
        "weighted": "#E8913A",
        "hybridrank": "#4CAF50",
    }

    # A. Barras horizontales in-sample por objective
    fig, ax = plt.subplots(figsize=(10, 7))
    df_sorted = insample_df.sort_values("objective", ascending=True)
    colors = [family_colors[f] for f in df_sorted["strategy_family"]]
    ax.barh(df_sorted["strategy"], df_sorted["objective"], color=colors)
    ax.set_xlabel("Objective (0.4*nDCG + 0.3*Recall + 0.2*MAP + 0.1*F1) @ k=10")
    ax.set_title("Fase 2: Comparacion In-Sample de Estrategias")
    ax.set_xlim(0.5, 0.8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "phase2_final_insample_comparison.png")
    plt.close(fig)

    # B. LOO-CV comparison bars with error
    fig, ax = plt.subplots(figsize=(6, 5))
    strategies = loocv_df["strategy"].values
    means = loocv_df["mean_test_objective"].values
    stds = loocv_df["std_test_objective"].values
    bar_colors = ["#E8913A", "#4CAF50"]
    bars = ax.bar(strategies, means, yerr=stds, capsize=8, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Mean Test Objective (LOO-CV)")
    ax.set_title("LOO-CV: Estimacion No Sesgada")
    ax.set_ylim(0.5, 0.9)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "phase2_loocv_comparison.png")
    plt.close(fig)

    # C. In-sample vs LOO-CV
    fig, ax = plt.subplots(figsize=(7, 5))
    insample_opt = insample_df[insample_df["strategy"].isin(["weighted_optimized", "hybridrank_optimized"])]
    x = np.arange(2)
    width = 0.35
    insample_vals = [
        float(insample_opt[insample_opt["strategy"] == "weighted_optimized"]["objective"].iloc[0]),
        float(insample_opt[insample_opt["strategy"] == "hybridrank_optimized"]["objective"].iloc[0]),
    ]
    loocv_vals = list(loocv_df["mean_test_objective"].values)
    ax.bar(x - width / 2, insample_vals, width, label="In-Sample", color=["#E8913A", "#4CAF50"], alpha=0.8)
    ax.bar(x + width / 2, loocv_vals, width, label="LOO-CV", color=["#E8913A", "#4CAF50"], alpha=0.4, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["weighted_optimized", "hybridrank_optimized"])
    ax.set_ylabel("Objective")
    ax.set_title("In-Sample vs LOO-CV (Overfitting Gap)")
    ax.legend()
    ax.set_ylim(0.5, 0.85)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "phase2_insample_vs_loocv.png")
    plt.close(fig)

    # D. Barras pareadas por query
    fig, ax = plt.subplots(figsize=(12, 5))
    colors_paired = ["#4CAF50" if w == "hybridrank" else "#E53935" if w == "weighted" else "#BDBDBD"
                     for w in paired_df["winner"]]
    ax.bar(paired_df["query_id"], paired_df["difference"], color=colors_paired, edgecolor="none")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xlabel("Query ID")
    ax.set_ylabel("Difference (HybridRank - Weighted)")
    ax.set_title("Comparacion Pareada: HybridRank vs Weighted (LOO-CV held-out)")
    ax.tick_params(axis="x", rotation=45)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "paired_hybridrank_vs_weighted.png")
    plt.close(fig)

    # E. Heatmap de metricas
    fig, ax = plt.subplots(figsize=(10, 7))
    metric_cols = ["recall", "precision", "f1", "mrr", "map", "ndcg", "objective"]
    heatmap_data = insample_df.set_index("strategy")[metric_cols]
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax,
                linewidths=0.5, cbar_kws={"label": "Score"})
    ax.set_title("Fase 2: Metricas Finales por Estrategia (In-Sample, k=10)")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "phase2_final_metrics_heatmap.png")
    plt.close(fig)

    print("  5 figuras generadas.")


def main():
    print("=" * 70)
    print("  FASE 2 - EVALUACION FINAL Y CIERRE")
    print("=" * 70)

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    queries = load_evaluation_data()
    print(f"Queries: {len(queries)}")

    metrics_list = [RecallAtK(), PrecisionAtK(), F1AtK(), MRR(), MAP(), NDCG()]

    # Inicializar retrievers y cache
    print("\nInicializando retrievers...")
    bm25 = BM25Retriever(index_dir=str(project_root / ".data" / "bm25_norma_index"))
    dense = DenseRetriever(
        chroma_dir=str(project_root / ".data" / "chroma_normas"),
        collection_name="hybridrank_normas",
    )

    candidate_k_values = sorted(set(c["candidate_k"] for c in STRATEGIES_FINAL.values()))
    print(f"Cacheando retrieval (candidate_k={candidate_k_values})...")
    cache = cache_retrieval_results(bm25, dense, queries, candidate_k_values)

    # 1. Evaluacion in-sample
    print("\n--- Evaluacion In-Sample ---")
    insample_df = run_insample_evaluation(queries, cache, metrics_list)
    insample_df.to_csv(OUTPUT_DIR / "final_phase2_insample.csv", index=False, encoding="utf-8")
    insample_df.to_csv(OUTPUT_DIR / "final_phase2_comparison.csv", index=False, encoding="utf-8")
    print(f"Guardado: final_phase2_insample.csv ({len(insample_df)} filas)")

    # 2. Resumen LOO-CV
    print("\n--- Resumen LOO-CV ---")
    loocv_df = build_loocv_summary()
    loocv_df.to_csv(OUTPUT_DIR / "final_phase2_loocv.csv", index=False, encoding="utf-8")
    print(loocv_df.to_string(index=False))

    # 3. Comparacion pareada
    print("\n--- Comparacion Pareada (held-out) ---")
    paired_df, summary = run_paired_comparison()
    paired_df.to_csv(OUTPUT_DIR / "paired_hybridrank_vs_weighted.csv", index=False, encoding="utf-8")
    with open(OUTPUT_DIR / "paired_hybridrank_vs_weighted_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  HybridRank wins: {summary['hybridrank_wins']}")
    print(f"  Weighted wins:   {summary['weighted_wins']}")
    print(f"  Ties:            {summary['ties']}")
    print(f"  Mean difference: {summary['mean_difference']}")

    # 4. Tests estadisticos
    print("\n--- Tests Estadisticos ---")
    stats = run_statistical_tests(paired_df)
    with open(OUTPUT_DIR / "paired_statistical_tests.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"  Sign test p-value: {stats['sign_test']['p_value']}")
    print(f"  Bootstrap CI 95%: [{stats['bootstrap']['ci_lower']}, {stats['bootstrap']['ci_upper']}]")
    print(f"  Wilcoxon p-value: {stats['wilcoxon']['p_value']}")

    # 5. Visualizaciones
    print("\n--- Visualizaciones ---")
    generate_visualizations(insample_df, loocv_df, paired_df)

    # Resumen final
    print("\n" + "=" * 70)
    print("  FASE 2 CERRADA")
    print("=" * 70)
    print(f"\n  Configuracion congelada:")
    print(f"    hybridrank_optimized: alpha=0.7, beta=0.2, k=10, minmax, candidate_k=50")
    print(f"    weighted_optimized:   alpha=0.7, minmax, candidate_k=20")
    print(f"\n  In-sample objective:")
    hr_obj = insample_df[insample_df["strategy"] == "hybridrank_optimized"]["objective"].iloc[0]
    w_obj = insample_df[insample_df["strategy"] == "weighted_optimized"]["objective"].iloc[0]
    print(f"    hybridrank_optimized: {hr_obj}")
    print(f"    weighted_optimized:   {w_obj}")
    print(f"\n  LOO-CV test objective:")
    print(f"    hybridrank_optimized: {loocv_df[loocv_df['strategy']=='hybridrank_optimized']['mean_test_objective'].iloc[0]}")
    print(f"    weighted_optimized:   {loocv_df[loocv_df['strategy']=='weighted_optimized']['mean_test_objective'].iloc[0]}")


if __name__ == "__main__":
    main()
