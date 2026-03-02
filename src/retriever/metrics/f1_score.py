# F1@k metric implementation
from typing import List, Tuple, Dict, Any, Optional
from .base import Metric


class F1AtK(Metric):
    """
    Implementación de F1@k.
    
    F1@k es la media armónica entre Precision@k y Recall@k:
    
    F1@k = 2 * (Precision@k * Recall@k) / (Precision@k + Recall@k)
    
    donde:
    - Precision@k = (# relevantes en top-k) / k
    - Recall@k = (# relevantes en top-k) / (# relevantes totales)
    
    F1@k proporciona un balance entre precisión (evitar falsos positivos)
    y cobertura (evitar falsos negativos), siendo especialmente útil cuando
    se necesita optimizar ambos objetivos simultáneamente.
    
    En RAG, F1@k ayuda a encontrar el valor óptimo de k que balancea:
    - Dar suficiente contexto al LLM (alto Recall)
    - Evitar ruido e información irrelevante (alta Precision)
    """
    
    def compute(
        self,
        retrieved_documents: List[Tuple[str, float]],
        relevant_documents: List[str],
        k: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calcula F1@k.
        
        Args:
            retrieved_documents: Lista de tuplas (doc_id, score) ordenadas por relevancia.
            relevant_documents: Lista de doc_ids relevantes.
            k: Número de documentos top a considerar. Si es None, usa todos los recuperados.
            **kwargs: Parámetros adicionales (no usados en esta métrica).
        
        Returns:
            Dict con:
                - 'score': float - Valor de F1@k en [0, 1]
                - 'metric_name': str - "F1@k" o "F1" si k es None
                - 'k': int - Valor de k usado
                - 'precision': float - Precision@k calculada
                - 'recall': float - Recall@k calculada
                - 'relevant_found': int - Número de docs relevantes en top-k
                - 'total_relevant': int - Número total de docs relevantes
                - 'total_retrieved': int - Número de docs en top-k (k)
        """
        if not retrieved_documents or not relevant_documents:
            effective_k = k if k is not None else (len(retrieved_documents) if retrieved_documents else 0)
            return {
                'score': 0.0,
                'metric_name': f'F1@{effective_k}' if k is not None else 'F1',
                'k': effective_k,
                'precision': 0.0,
                'recall': 0.0,
                'relevant_found': 0,
                'total_relevant': len(relevant_documents) if relevant_documents else 0,
                'total_retrieved': effective_k
            }
        
        if k is None:
            k_effective = len(retrieved_documents)
        else:
            k_effective = min(k, len(retrieved_documents))
        
        if k_effective == 0:
            return {
                'score': 0.0,
                'metric_name': 'F1@0',
                'k': 0,
                'precision': 0.0,
                'recall': 0.0,
                'relevant_found': 0,
                'total_relevant': len(relevant_documents),
                'total_retrieved': 0
            }
        
        top_k_ids = {doc_id for doc_id, _ in retrieved_documents[:k_effective]}
        relevant_set = set(relevant_documents)
        relevant_found = len(top_k_ids.intersection(relevant_set))
        precision = relevant_found / k_effective
        
        recall = relevant_found / len(relevant_documents)
        
        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)
        
        return {
            'score': f1_score,
            'metric_name': f'F1@{k_effective}' if k is not None else 'F1',
            'k': k_effective,
            'precision': precision,
            'recall': recall,
            'relevant_found': relevant_found,
            'total_relevant': len(relevant_documents),
            'total_retrieved': k_effective
        }
    
    @property
    def name(self) -> str:
        """Retorna el nombre de la métrica."""
        return "F1@k"
