# Recall@k metric implementation
from typing import List, Tuple, Dict, Any, Optional
from .base import Metric


class RecallAtK(Metric):
    """
    Implementación de Recall@k.
    
    Recall@k mide la proporción de documentos relevantes que aparecen
    en los top-k documentos recuperados:
    
    Recall@k = (# docs relevantes en top-k) / (# docs relevantes totales)
    
    En RAG Recall@k mide la probabilidad de que al menos un pasaje útil
    esté disponible para el LLM.
    """
    
    def compute(
        self,
        retrieved_documents: List[Tuple[str, float]],
        relevant_documents: List[str],
        k: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calcula Recall@k.
        
        Args:
            retrieved_documents: Lista de tuplas (doc_id, score) ordenadas por relevancia.
            relevant_documents: Lista de doc_ids relevantes.
            k: Número de documentos top a considerar. Si es None, usa todos los recuperados.
            **kwargs: Parámetros adicionales (no usados en esta métrica).
        
        Returns:
            Dict con:
                - 'score': float - Valor de Recall@k en [0, 1]
                - 'metric_name': str - "Recall@k" o "Recall" si k es None
                - 'k': int - Valor de k usado
                - 'relevant_found': int - Número de docs relevantes encontrados en top-k
                - 'total_relevant': int - Número total de docs relevantes
        """
        if not relevant_documents:
            effective_k = k if k is not None else len(retrieved_documents)
            return {
                'score': 0.0,
                'metric_name': f'Recall@{effective_k}' if k is not None else 'Recall',
                'k': effective_k,
                'relevant_found': 0,
                'total_relevant': 0
            }
        
        if k is None:
            k_effective = len(retrieved_documents)
        else:
            k_effective = min(k, len(retrieved_documents))
        
        top_k_ids = {doc_id for doc_id, _ in retrieved_documents[:k_effective]}
        relevant_set = set(relevant_documents)
        relevant_found = len(top_k_ids.intersection(relevant_set))
        
        recall_score = relevant_found / len(relevant_documents)
        
        return {
            'score': recall_score,
            'metric_name': f'Recall@{k_effective}' if k is not None else 'Recall',
            'k': k_effective,
            'relevant_found': relevant_found,
            'total_relevant': len(relevant_documents)
        }
    
    @property
    def name(self) -> str:
        """Retorna el nombre de la métrica."""
        return "Recall@k"
