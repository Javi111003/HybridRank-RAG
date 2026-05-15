from abc import ABC, abstractmethod
from typing import Dict


class ScoreNormalizer(ABC):
    """Interfaz abstracta para normalizadores de scores."""

    @abstractmethod
    def normalize(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Normaliza un diccionario de scores.

        Args:
            scores: Dict doc_id -> score original.

        Returns:
            Dict doc_id -> score normalizado.
        """
        pass


class MinMaxNormalizer(ScoreNormalizer):
    """
    Normalización Min-Max: escala scores al rango [0, 1].

    Fórmula: (score - min) / (max - min)

    Caso borde: si todos los scores son iguales (max == min),
    retorna 1.0 para todos los documentos.
    """

    def normalize(self, scores: Dict[str, float]) -> Dict[str, float]:
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
    """
    Normalización Z-Score: centra en media 0 y desviación estándar 1.

    Fórmula: (score - mean) / std

    Caso borde: si todos los scores son iguales (std == 0),
    retorna 0.0 para todos los documentos.
    """

    def normalize(self, scores: Dict[str, float]) -> Dict[str, float]:
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
    """
    Normalización por suma: divide cada score por la suma total.

    Fórmula: score / sum(scores)

    Caso borde: si la suma es 0, retorna 0.0 para todos.
    Nota: Requiere que los scores sean no-negativos para producir
    resultados en [0, 1].
    """

    def normalize(self, scores: Dict[str, float]) -> Dict[str, float]:
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
    """Normalizador identidad: retorna los scores sin cambios."""

    def normalize(self, scores: Dict[str, float]) -> Dict[str, float]:
        return dict(scores)


def get_normalizer(name: str) -> ScoreNormalizer:
    """
    Factory de normalizadores.

    Args:
        name: Nombre del normalizador: "minmax", "zscore", "sum", "identity".

    Returns:
        Instancia de ScoreNormalizer.

    Raises:
        ValueError: Si el nombre no es reconocido.
    """
    normalizers = {
        "minmax": MinMaxNormalizer,
        "zscore": ZScoreNormalizer,
        "sum": SumNormalizer,
        "identity": IdentityNormalizer,
    }

    if name not in normalizers:
        raise ValueError(
            f"Normalizador desconocido: '{name}'. "
            f"Disponibles: {list(normalizers.keys())}"
        )

    return normalizers[name]()
