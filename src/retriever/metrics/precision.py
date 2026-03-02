# Precision@k metric implementation
from typing import List, Tuple, Dict, Any, Optional
from .base import Metric


class PrecisionAtK(Metric):
    """
    Implementación de Precision@k.
    
    Precision@k mide la proporción de documentos relevantes dentro de los
    top-k documentos recuperados:
    
    Precision@k = (# docs relevantes en top-k) / k
    
    A diferencia de Recall@k (que mide cobertura), Precision@k mide la
    "limpieza" o "pureza" del ranking. Un Precision@k alto significa que
    el sistema evita incluir documentos irrelevantes en el top-k.
    
    En RAG, Precision@k alto reduce el ruido en el contexto proporcionado
    al LLM, evitando distracciones con información irrelevante que podrían
    degradar la calidad de la respuesta generada.
    """
    
    def compute(
        self,
        retrieved_documents: List[Tuple[str, float]],
        relevant_documents: List[str],
        k: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calcula Precision@k.
        
        Args:
            retrieved_documents: Lista de tuplas (doc_id, score) ordenadas por relevancia.
            relevant_documents: Lista de doc_ids relevantes.
            k: Número de documentos top a considerar. Si es None, usa todos los recuperados.
            **kwargs: Parámetros adicionales (no usados en esta métrica).
        
        Returns:
            Dict con:
                - 'score': float - Valor de Precision@k en [0, 1]
                - 'metric_name': str - "Precision@k" o "Precision" si k es None
                - 'k': int - Valor de k usado
                - 'relevant_found': int - Número de docs relevantes encontrados en top-k
                - 'total_retrieved': int - Número de docs en top-k (igual a k)
        """
        if not retrieved_documents:
            effective_k = k if k is not None else 0
            return {
                'score': 0.0,
                'metric_name': f'Precision@{effective_k}' if k is not None else 'Precision',
                'k': effective_k,
                'relevant_found': 0,
                'total_retrieved': 0
            }
        
        if not relevant_documents:
            effective_k = k if k is not None else len(retrieved_documents)
            return {
                'score': 0.0,
                'metric_name': f'Precision@{effective_k}' if k is not None else 'Precision',
                'k': effective_k,
                'relevant_found': 0,
                'total_retrieved': effective_k
            }
        
        if k is None:
            k_effective = len(retrieved_documents)
        else:
            k_effective = min(k, len(retrieved_documents))
        
        if k_effective == 0:
            return {
                'score': 0.0,
                'metric_name': 'Precision@0',
                'k': 0,
                'relevant_found': 0,
                'total_retrieved': 0
            }
        
        top_k_ids = {doc_id for doc_id, _ in retrieved_documents[:k_effective]}
        relevant_set = set(relevant_documents)
        relevant_found = len(top_k_ids.intersection(relevant_set))
        
        precision_score = relevant_found / k_effective
        
        return {
            'score': precision_score,
            'metric_name': f'Precision@{k_effective}' if k is not None else 'Precision',
            'k': k_effective,
            'relevant_found': relevant_found,
            'total_retrieved': k_effective
        }
    
    @property
    def name(self) -> str:
        """Retorna el nombre de la métrica."""
        return "Precision@k"
