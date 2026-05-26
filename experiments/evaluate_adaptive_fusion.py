"""
Evaluación Leave-One-Out de sistemas adaptativos de fusión.

Compara 4 sistemas:
1. global_hybridrank_optimized: Mejor config global fija (baseline)
2. oracle_adaptive: Adaptive con query_type real (upper bound)
3. rule_based_adaptive: Adaptive con clasificador basado en reglas
4. llm_adaptive: Adaptive con clasificador LLM

Uso:
    .venv/Scripts/python.exe experiments/evaluate_adaptive_fusion.py
    .venv/Scripts/python.exe experiments/evaluate_adaptive_fusion.py --skip-llm

Outputs:
    experiments/results/adaptive_fusion_loocv_results.csv
    experiments/results/adaptive_fusion_summary.csv
    experiments/results/query_classification_results.csv
    experiments/results/query_classification_summary.json
    experiments/results/query_classification_confusion_matrix.csv
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retriever import BM25Retriever, DenseRetriever
from src.retriever.fusion.registry import get_fusion_strategy
from src.retriever.metrics import RecallAtK, PrecisionAtK, F1AtK, MRR, MAP, NDCG
from src.adaptive import (
    QuerySignalExtractor,
    QueryClassification,
    OracleQueryClassifier,
    RuleBasedQueryClassifier,
    LLMQueryClassifier,
    HybridQueryClassifier,
    AdaptiveFusionPolicy,
    FusionConfig,
    GLOBAL_FALLBACK,
    compute_objective,
    compute_metrics_for_query,
    build_search_space,
    QUERY_TYPES,
)

EVALUATION_DATA_PATH = project_root / ".data" / "evaluation" / "norma_qrels.json"
OUTPUT_DIR = project_root / "experiments" / "results"
CANDIDATE_K_VALUES = [20, 50, 100]
OBJECTIVE_K = 10

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
        description="Evaluación LOO-CV de fusión adaptativa"
    )
    parser.add_argument(
        "--skip-llm", action="store_true",
        help="Omitir evaluación LLM (no requiere API keys)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Progreso detallado"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directorio de salida"
    )
    return parser.parse_args()


def load_evaluation_data() -> List[Dict[str, Any]]:
    with open(EVALUATION_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    queries = [q for q in data if q.get("relevant_docs")]
    print(f"Cargadas {len(queries)} queries con relevancia")
    return queries


def cache_retrieval_results(
    bm25: BM25Retriever,
    dense: DenseRetriever,
    queries: List[Dict[str, Any]],
    candidate_k_values: List[int],
) -> Dict[int, Dict[str, Dict[str, List[Tuple[str, float]]]]]:
    """Pre-recupera resultados de BM25 y Dense para cada candidate_k."""
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


def evaluate_system_loocv(
    system_name: str,
    queries: List[Dict[str, Any]],
    cache: Dict[int, Dict[str, Dict[str, List[Tuple[str, float]]]]],
    metrics: List[Any],
    classifier_factory,
    search_space: List[FusionConfig],
    signal_extractor: QuerySignalExtractor,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Evalúa un sistema adaptativo con LOO-CV.

    Para global_hybridrank_optimized: classifier_factory=None, se usa GLOBAL_FALLBACK fijo.
    Para sistemas adaptativos: classifier_factory devuelve un QueryClassifier.
    """
    n = len(queries)
    rows = []

    for i in tqdm(range(n), desc=f"LOO-CV {system_name}", total=n):
        held_out = queries[i]
        train = [q for j, q in enumerate(queries) if j != i]
        held_out_qid = held_out["query_id"]
        held_out_type = held_out["query_type"]

        if classifier_factory is None:
            # Sistema global: aplica GLOBAL_FALLBACK directo
            config = GLOBAL_FALLBACK
            predicted_type = "global"
            confidence = 1.0
            reason = "configuración global fija"
            fallback_used = False
        else:
            classifier = classifier_factory()

            # Clasificar queries de train
            train_type_assignments: Dict[str, str] = {}
            for q in train:
                if isinstance(classifier, OracleQueryClassifier):
                    cl = classifier.classify_by_id(q["query_id"])
                else:
                    cl = classifier.classify(q["query"])
                    # Delay proactivo para LLM para evitar rate limits
                    if isinstance(classifier, LLMQueryClassifier):
                        time.sleep(0.3)
                train_type_assignments[q["query_id"]] = cl.query_type

            # Entrenar política
            policy = AdaptiveFusionPolicy(
                search_space=search_space,
                global_fallback=GLOBAL_FALLBACK,
                min_train_examples_per_type=3,
            )
            policy.fit(train, cache, train_type_assignments)

            # Clasificar held-out
            if isinstance(classifier, OracleQueryClassifier):
                classification = classifier.classify_by_id(held_out_qid)
            else:
                signals = signal_extractor.extract_static(held_out["query"])
                classification = classifier.classify(held_out["query"], signals)
                # Delay proactivo para LLM para evitar rate limits
                if isinstance(classifier, LLMQueryClassifier):
                    time.sleep(0.3)

            predicted_type = classification.query_type
            confidence = classification.confidence
            reason = classification.reason

            # Seleccionar config
            config = policy.select_config(predicted_type, confidence)
            fallback_used = config == GLOBAL_FALLBACK

        # Aplicar fusión y evaluar
        ck = config.candidate_k
        results_by_retriever = cache[ck][held_out_qid]
        strategy = config.build_strategy()
        fused = strategy.fuse(results_by_retriever, top_k=20)

        m = compute_metrics_for_query(
            fused, held_out["relevant_docs"], metrics, OBJECTIVE_K
        )
        obj = compute_objective(m)

        # Señales de retrieval
        bm25_res = cache[50][held_out_qid]["bm25"]
        dense_res = cache[50][held_out_qid]["dense"]
        bm25_ids = {d for d, _ in bm25_res[:10]}
        dense_ids = {d for d, _ in dense_res[:10]}
        overlap_at_10 = len(bm25_ids & dense_ids) / 10.0

        row = {
            "query_id": held_out_qid,
            "query_type_gold": held_out_type,
            "system": system_name,
            "predicted_query_type": predicted_type,
            "classifier_confidence": round(confidence, 4),
            "classifier_reason": reason,
            "selected_strategy": config.strategy_name,
            "selected_params": config.params_json(),
            "candidate_k": config.candidate_k,
            "fallback_used": fallback_used,
            "overlap_at_10": round(overlap_at_10, 4),
            "top1_bm25_score": round(bm25_res[0][1], 4) if bm25_res else 0.0,
            "top1_dense_score": round(dense_res[0][1], 4) if dense_res else 0.0,
            "recall": round(m.get("recall", 0.0), 4),
            "precision": round(m.get("precision", 0.0), 4),
            "f1": round(m.get("f1", 0.0), 4),
            "mrr": round(m.get("mrr", 0.0), 4),
            "map": round(m.get("map", 0.0), 4),
            "ndcg": round(m.get("ndcg", 0.0), 4),
            "objective": round(obj, 4),
        }
        rows.append(row)

        if verbose:
            print(
                f"  {held_out_qid} [{held_out_type}] -> pred={predicted_type} "
                f"obj={obj:.4f} fallback={fallback_used}"
            )

    return rows


def evaluate_classification(
    queries: List[Dict[str, Any]],
    classifiers: Dict[str, Any],
    signal_extractor: QuerySignalExtractor,
) -> pd.DataFrame:
    """Evalúa accuracy de clasificadores contra ground truth."""
    rows = []
    for q in queries:
        row = {
            "query_id": q["query_id"],
            "query": q["query"],
            "gold_query_type": q["query_type"],
        }
        for clf_name, clf in classifiers.items():
            if clf is None:
                row[f"{clf_name}_pred"] = ""
                row[f"{clf_name}_confidence"] = 0.0
                row[f"correct_{clf_name}"] = False
                continue
            signals = signal_extractor.extract_static(q["query"])
            result = clf.classify(q["query"], signals)
            row[f"{clf_name}_pred"] = result.query_type
            row[f"{clf_name}_confidence"] = round(result.confidence, 4)
            row[f"correct_{clf_name}"] = result.query_type == q["query_type"]
        rows.append(row)
    return pd.DataFrame(rows)


def build_confusion_matrix(
    df: pd.DataFrame, pred_col: str
) -> pd.DataFrame:
    """Construye matriz de confusión desde el DataFrame de clasificación."""
    types = sorted(QUERY_TYPES)
    matrix = pd.DataFrame(0, index=types, columns=types)
    for _, row in df.iterrows():
        gold = row["gold_query_type"]
        pred = row[pred_col]
        if gold in types and pred in types:
            matrix.loc[gold, pred] += 1
    return matrix


def build_classification_summary(
    df: pd.DataFrame, classifier_names: List[str]
) -> Dict[str, Any]:
    """Resume accuracy y per-type accuracy para cada clasificador."""
    summary: Dict[str, Any] = {}
    for name in classifier_names:
        correct_col = f"correct_{name}"
        if correct_col not in df.columns:
            continue
        valid = df[df[f"{name}_pred"] != ""]
        if len(valid) == 0:
            continue
        accuracy = valid[correct_col].mean()
        per_type = {}
        for qt in QUERY_TYPES:
            subset = valid[valid["gold_query_type"] == qt]
            if len(subset) > 0:
                per_type[qt] = round(float(subset[correct_col].mean()), 4)
        summary[name] = {
            "accuracy": round(float(accuracy), 4),
            "n_queries": int(len(valid)),
            "per_type_accuracy": per_type,
        }
    return summary


def main():
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EVALUACIÓN ADAPTATIVA LOO-CV")
    print("=" * 60)

    # Cargar datos
    queries = load_evaluation_data()
    metrics = [RecallAtK(), PrecisionAtK(), F1AtK(), MRR(), MAP(), NDCG()]
    signal_extractor = QuerySignalExtractor()

    # Inicializar retrievers y cachear
    print("\nInicializando retrievers...")
    bm25 = BM25Retriever(index_dir=str(project_root / ".data" / "bm25_norma_index"))
    dense = DenseRetriever(
        chroma_dir=str(project_root / ".data" / "chroma_normas"),
        collection_name="hybridrank_normas",
    )

    print("Cacheando resultados de retrieval...")
    t0 = time.time()
    cache = cache_retrieval_results(bm25, dense, queries, CANDIDATE_K_VALUES)
    print(f"  Cache completado en {time.time() - t0:.1f}s")

    # Construir espacio de búsqueda
    search_space = build_search_space()
    print(f"Espacio de búsqueda: {len(search_space)} configuraciones")

    # Clasificadores
    query_type_map = {q["query_id"]: q["query_type"] for q in queries}
    oracle = OracleQueryClassifier(query_type_map)
    rule_based = RuleBasedQueryClassifier()

    llm_classifier = None
    hybrid_classifier = None
    if not args.skip_llm:
        try:
            from src.rag.generator.registry import get_generator
            generator = get_generator("mistral", temperature=0.0)
            llm_classifier = LLMQueryClassifier(generator)
            hybrid_classifier = HybridQueryClassifier(rule_based, llm_classifier)
            print("LLM classifier inicializado correctamente")
        except Exception as e:
            print(f"ERROR: No se pudo inicializar LLM classifier: {e}")
            print("Sugerencia: usar --skip-llm para evaluar sin LLM")
            sys.exit(1)

    # Definir sistemas
    systems = [
        ("global_hybridrank_optimized", None),
        ("oracle_adaptive", lambda: oracle),
        ("rule_based_adaptive", lambda: rule_based),
    ]
    if llm_classifier:
        systems.append(("llm_adaptive", lambda: llm_classifier))

    # Ejecutar LOO-CV
    all_results: List[Dict[str, Any]] = []
    for system_name, clf_factory in systems:
        print(f"\n{'-' * 40}")
        print(f"Evaluando: {system_name}")
        print(f"{'-' * 40}")
        t0 = time.time()
        rows = evaluate_system_loocv(
            system_name=system_name,
            queries=queries,
            cache=cache,
            metrics=metrics,
            classifier_factory=clf_factory,
            search_space=search_space,
            signal_extractor=signal_extractor,
            verbose=args.verbose,
        )
        elapsed = time.time() - t0
        mean_obj = np.mean([r["objective"] for r in rows])
        print(f"  Completado en {elapsed:.1f}s | mean_objective={mean_obj:.4f}")
        all_results.extend(rows)

    # Guardar resultados LOO-CV
    df_results = pd.DataFrame(all_results)
    results_path = output_dir / "adaptive_fusion_loocv_results.csv"
    df_results.to_csv(results_path, index=False)
    print(f"\nResultados LOO-CV guardados: {results_path}")

    # Guardar summary
    summary_rows = []
    for system_name, _ in systems:
        sys_df = df_results[df_results["system"] == system_name]
        summary_rows.append({
            "system": system_name,
            "mean_recall": round(float(sys_df["recall"].mean()), 4),
            "mean_precision": round(float(sys_df["precision"].mean()), 4),
            "mean_f1": round(float(sys_df["f1"].mean()), 4),
            "mean_mrr": round(float(sys_df["mrr"].mean()), 4),
            "mean_map": round(float(sys_df["map"].mean()), 4),
            "mean_ndcg": round(float(sys_df["ndcg"].mean()), 4),
            "mean_objective": round(float(sys_df["objective"].mean()), 4),
            "std_objective": round(float(sys_df["objective"].std()), 4),
        })
    df_summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "adaptive_fusion_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"Summary guardado: {summary_path}")

    # Evaluación de clasificación
    print(f"\n{'-' * 40}")
    print("Evaluando clasificadores")
    print(f"{'-' * 40}")

    classifiers_to_eval = {"rule_based": rule_based}
    if llm_classifier:
        classifiers_to_eval["llm"] = llm_classifier
    if hybrid_classifier:
        classifiers_to_eval["hybrid"] = hybrid_classifier

    df_clf = evaluate_classification(queries, classifiers_to_eval, signal_extractor)
    clf_path = output_dir / "query_classification_results.csv"
    df_clf.to_csv(clf_path, index=False)
    print(f"Clasificación guardada: {clf_path}")

    # Summary de clasificación
    clf_summary = build_classification_summary(df_clf, list(classifiers_to_eval.keys()))
    summary_json_path = output_dir / "query_classification_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(clf_summary, f, indent=2, ensure_ascii=False)
    print(f"Summary clasificación: {summary_json_path}")

    # Matriz de confusión
    for clf_name in classifiers_to_eval:
        pred_col = f"{clf_name}_pred"
        if pred_col in df_clf.columns:
            cm = build_confusion_matrix(df_clf, pred_col)
            cm_path = output_dir / f"query_classification_confusion_matrix_{clf_name}.csv"
            cm.to_csv(cm_path)
            print(f"Matriz de confusión ({clf_name}): {cm_path}")

    # Imprimir resumen final
    print(f"\n{'=' * 60}")
    print("RESUMEN FINAL")
    print(f"{'=' * 60}")
    print(df_summary.to_string(index=False))

    if clf_summary:
        print(f"\nAccuracy de clasificadores:")
        for name, info in clf_summary.items():
            print(f"  {name}: {info['accuracy']:.4f} ({info['n_queries']} queries)")

    # Gaps
    global_obj = df_summary[df_summary["system"] == "global_hybridrank_optimized"]["mean_objective"].values[0]
    for _, row in df_summary.iterrows():
        if row["system"] != "global_hybridrank_optimized":
            gap = row["mean_objective"] - global_obj
            print(f"  {row['system']} vs global: {gap:+.4f}")


if __name__ == "__main__":
    main()
