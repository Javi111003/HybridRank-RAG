from typing import Dict, List, Set, Tuple


def results_to_score_dict(results: List[Tuple[str, float]]) -> Dict[str, float]:
    """Convierte lista de resultados a dict doc_id -> score."""
    return {doc_id: score for doc_id, score in results}


def results_to_rank_dict(results: List[Tuple[str, float]]) -> Dict[str, int]:
    """Convierte lista de resultados a dict doc_id -> rank (1-indexed)."""
    return {doc_id: rank for rank, (doc_id, _) in enumerate(results, start=1)}


def get_all_doc_ids(
    results_by_retriever: Dict[str, List[Tuple[str, float]]],
) -> Set[str]:
    """Obtiene la unión de todos los doc_ids recuperados por todos los retrievers."""
    all_ids: Set[str] = set()
    for results in results_by_retriever.values():
        for doc_id, _ in results:
            all_ids.add(doc_id)
    return all_ids


def sort_and_truncate(
    scores: Dict[str, float], top_k: int
) -> List[Tuple[str, float]]:
    """Ordena docs por score descendente y trunca a top_k."""
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]


def fill_missing_scores(
    score_dict: Dict[str, float],
    all_docs: Set[str],
    default: float = 0.0,
) -> Dict[str, float]:
    """Rellena documentos ausentes con un score por defecto."""
    filled = dict(score_dict)
    for doc_id in all_docs:
        if doc_id not in filled:
            filled[doc_id] = default
    return filled
