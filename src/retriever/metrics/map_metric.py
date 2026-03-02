# Mean Average Precision (MAP) metric implementation
from typing import List, Tuple, Dict, Any, Optional
from .base import Metric


class MAP(Metric):
    """
    Implementación de Mean Average Precision (MAP).
    
    MAP calcula el promedio de las precisiones en cada posición donde
    aparece un documento relevante:
    
    Average Precision = (1/|relevantes|) * Σ(Precision@i * rel(i))
    
    donde:
    - Precision@i es la precisión en la posición i
    - rel(i) = 1 si el documento en posición i es relevante, 0 en otro caso
    
    MAP es el promedio de AP sobre múltiples queries. En este caso,
    calculamos AP para una sola query (equivalente a Average Precision).
    """
    
    def compute(
        self,
        retrieved_documents: List[Tuple[str, float]],
        relevant_documents: List[str],
        k: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calcula MAP (Mean Average Precision).
        
        Args:
            retrieved_documents: Lista de tuplas (doc_id, score) ordenadas por relevancia.
            relevant_documents: Lista de doc_ids relevantes.
            k: Opcional. Si se especifica, solo considera los top-k documentos (MAP@k).
            **kwargs: Parámetros adicionales (no usados en esta métrica).
        
        Returns:
            Dict con:
                - 'score': float - Valor de MAP en [0, 1]
                - 'metric_name': str - "MAP@k" o "MAP"
                - 'average_precision': float - Average Precision para esta query
                - 'precisions_at_relevant': List[float] - Precisiones en posiciones relevantes
                - 'num_relevant_retrieved': int - Número de relevantes recuperados
        """
        if not relevant_documents:
            metric_name = f'MAP@{k}' if k is not None else 'MAP'
            return {
                'score': 0.0,
                'metric_name': metric_name,
                'average_precision': 0.0,
                'precisions_at_relevant': [],
                'num_relevant_retrieved': 0
            }
        
        if not retrieved_documents:
            metric_name = f'MAP@{k}' if k is not None else 'MAP'
            return {
                'score': 0.0,
                'metric_name': metric_name,
                'average_precision': 0.0,
                'precisions_at_relevant': [],
                'num_relevant_retrieved': 0
            }
        
        docs_to_check = retrieved_documents
        if k is not None:
            docs_to_check = retrieved_documents[:k]
        
        relevant_set = set(relevant_documents)
        precisions_at_relevant = []
        num_relevant_seen = 0
        
        for rank, (doc_id, _) in enumerate(docs_to_check, start=1):
            if doc_id in relevant_set:
                num_relevant_seen += 1
                precision_at_rank = num_relevant_seen / rank
                precisions_at_relevant.append(precision_at_rank)
        
        if precisions_at_relevant:
            average_precision = sum(precisions_at_relevant) / len(relevant_documents)
        else:
            average_precision = 0.0
        
        metric_name = f'MAP@{k}' if k is not None else 'MAP'
        
        return {
            'score': average_precision,
            'metric_name': metric_name,
            'average_precision': average_precision,
            'precisions_at_relevant': precisions_at_relevant,
            'num_relevant_retrieved': num_relevant_seen
        }
    
    @property
    def name(self) -> str:
        """Retorna el nombre de la métrica."""
        return "MAP"
