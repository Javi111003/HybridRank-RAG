from src.retriever.types import ResultsByRetriever, RetrievalResults


def require_results(
    results_by_retriever: ResultsByRetriever,
    key: str,
    label: str,
) -> RetrievalResults:
    try:
        return results_by_retriever[key]
    except KeyError as exc:
        raise ValueError(
            f"Recuperador {label} '{key}' no encontrado. "
            f"Disponibles: {list(results_by_retriever.keys())}"
        ) from exc


def results_to_score_dict(results: RetrievalResults) -> dict[str, float]:
    return {doc_id: score for doc_id, score in results}


def results_to_rank_dict(results: RetrievalResults) -> dict[str, int]:
    return {doc_id: rank for rank, (doc_id, _) in enumerate(results, start=1)}


def get_all_doc_ids(results_by_retriever: ResultsByRetriever) -> set[str]:
    return {
        doc_id
        for results in results_by_retriever.values()
        for doc_id, _ in results
    }


def sort_and_truncate(scores: dict[str, float], top_k: int) -> RetrievalResults:
    if top_k <= 0:
        return []
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]


def fill_missing_scores(
    score_dict: dict[str, float],
    all_docs: set[str],
    default: float = 0.0,
) -> dict[str, float]:
    return {doc_id: score_dict.get(doc_id, default) for doc_id in all_docs}
