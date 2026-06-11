from abc import ABC, abstractmethod


class ScoreNormalizer(ABC):
    """Score normalization strategy."""

    @abstractmethod
    def normalize(self, scores: dict[str, float]) -> dict[str, float]:
        raise NotImplementedError


class MinMaxNormalizer(ScoreNormalizer):
    def normalize(self, scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}

        values = list(scores.values())
        min_val = min(values)
        max_val = max(values)

        if max_val == min_val:
            return {doc_id: 1.0 for doc_id in scores}

        range_val = max_val - min_val
        return {
            doc_id: (score - min_val) / range_val
            for doc_id, score in scores.items()
        }


class ZScoreNormalizer(ScoreNormalizer):
    def normalize(self, scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}

        values = list(scores.values())
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        std = variance ** 0.5

        if std == 0.0:
            return {doc_id: 0.0 for doc_id in scores}

        return {
            doc_id: (score - mean) / std
            for doc_id, score in scores.items()
        }


class SumNormalizer(ScoreNormalizer):
    def normalize(self, scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}

        total = sum(scores.values())

        if total == 0.0:
            return {doc_id: 0.0 for doc_id in scores}

        return {
            doc_id: score / total
            for doc_id, score in scores.items()
        }


class IdentityNormalizer(ScoreNormalizer):
    def normalize(self, scores: dict[str, float]) -> dict[str, float]:
        return dict(scores)


NORMALIZERS = {
    "minmax": MinMaxNormalizer,
    "zscore": ZScoreNormalizer,
    "sum": SumNormalizer,
    "identity": IdentityNormalizer,
}


def get_normalizer(name: str) -> ScoreNormalizer:
    normalizer_cls = NORMALIZERS.get(name)
    if normalizer_cls is None:
        raise ValueError(
            f"Normalizador desconocido: '{name}'. "
            f"Disponibles: {list(NORMALIZERS.keys())}"
        )

    return normalizer_cls()
