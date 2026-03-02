# Normalized Discounted Cumulative Gain (nDCG) metric implementation
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from .base import Metric


class NDCG(Metric):
    """
    Implementación de Normalized Discounted Cumulative Gain (nDCG@k).
    
    nDCG mide la calidad del ranking considerando tanto la relevancia de los
    documentos como su posición en el ranking:
    
    DCG@k = Σ(i=1 to k) [(2^rel_i - 1) / log2(i + 1)]
    nDCG@k = DCG@k / IDCG@k
    
    donde:
    - rel_i es la relevancia del documento en posición i
    - IDCG@k es el DCG ideal (documentos perfectamente ordenados)
    
    En esta implementación usamos relevancia binaria (rel=1 si relevante, 0 si no).
    """
    
    def compute(
        self,
        retrieved_documents: List[Tuple[str, float]],
        relevant_documents: List[str],
        k: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calcula nDCG@k (Normalized Discounted Cumulative Gain).
        
        Args:
            retrieved_documents: Lista de tuplas (doc_id, score) ordenadas por relevancia.
            relevant_documents: Lista de doc_ids relevantes.
            k: Número de documentos top a considerar. Si es None, usa todos los recuperados.
            **kwargs: Parámetros adicionales (no usados en esta métrica).
        
        Returns:
            Dict con:
                - 'score': float - Valor de nDCG@k en [0, 1]
                - 'metric_name': str - "nDCG@k"
                - 'k': int - Valor de k usado
                - 'dcg': float - Discounted Cumulative Gain
                - 'idcg': float - Ideal DCG
        """
        if not relevant_documents or not retrieved_documents:
            effective_k = k if k is not None else (len(retrieved_documents) if retrieved_documents else 0)
            return {
                'score': 0.0,
                'metric_name': f'nDCG@{effective_k}',
                'k': effective_k,
                'dcg': 0.0,
                'idcg': 0.0
            }
        
        if k is None:
            k_effective = len(retrieved_documents)
        else:
            k_effective = min(k, len(retrieved_documents))
        
        relevant_set = set(relevant_documents)
        
        dcg = 0.0
        for i, (doc_id, _) in enumerate(retrieved_documents[:k_effective], start=1):
            relevance = 1 if doc_id in relevant_set else 0
            dcg += (2**relevance - 1) / np.log2(i + 1)
        
        # Calcular IDCG@k (ideal DCG con ranking perfecto)
        # En el caso ideal, todos los documentos relevantes estarían primero
        num_relevant = len(relevant_documents)
        num_ideal_relevant = min(num_relevant, k_effective)
        
        idcg = 0.0
        for i in range(1, num_ideal_relevant + 1):
            # En el caso ideal, los primeros num_ideal_relevant docs son relevantes
            idcg += (2**1 - 1) / np.log2(i + 1)
        
        if idcg == 0.0:
            return {
                'score': 0.0,
                'metric_name': f'nDCG@{k_effective}',
                'k': k_effective,
                'dcg': dcg,
                'idcg': idcg
            }
        
        ndcg_score = dcg / idcg
        
        return {
            'score': ndcg_score,
            'metric_name': f'nDCG@{k_effective}',
            'k': k_effective,
            'dcg': dcg,
            'idcg': idcg
        }
    
    @property
    def name(self) -> str:
        """Retorna el nombre de la métrica."""
        return "nDCG@k"
