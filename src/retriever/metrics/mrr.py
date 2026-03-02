# Mean Reciprocal Rank (MRR) metric implementation
from typing import List, Tuple, Dict, Any, Optional
from .base import Metric


class MRR(Metric):
    """
    Implementación de Mean Reciprocal Rank (MRR).
    
    MRR mide la posición del primer documento relevante en el ranking:
    
    MRR = 1 / rank_del_primer_documento_relevante
    
    Donde rank empieza en 1 (el primer documento tiene rank=1).
    Si ningún documento es relevante, MRR = 0.
    
    Esta métrica es especialmente útil cuando solo importa encontrar
    al menos un documento relevante lo más arriba posible en el ranking.
    """
    
    def compute(
        self,
        retrieved_documents: List[Tuple[str, float]],
        relevant_documents: List[str],
        k: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calcula MRR (Mean Reciprocal Rank).
        
        Args:
            retrieved_documents: Lista de tuplas (doc_id, score) ordenadas por relevancia.
            relevant_documents: Lista de doc_ids relevantes.
            k: Opcional. Si se especifica, solo considera los top-k documentos.
            **kwargs: Parámetros adicionales (no usados en esta métrica).
        
        Returns:
            Dict con:
                - 'score': float - Valor de MRR en [0, 1]
                - 'metric_name': str - "MRR"
                - 'first_relevant_rank': int o None - Posición del primer relevante (1-indexed)
        """
        if not relevant_documents or not retrieved_documents:
            return {
                'score': 0.0,
                'metric_name': 'MRR',
                'first_relevant_rank': None
            }
        
        docs_to_check = retrieved_documents
        if k is not None:
            docs_to_check = retrieved_documents[:k]
        
        relevant_set = set(relevant_documents)
        
        for rank, (doc_id, _) in enumerate(docs_to_check, start=1):
            if doc_id in relevant_set:
                return {
                    'score': 1.0 / rank,
                    'metric_name': 'MRR',
                    'first_relevant_rank': rank
                }
        return {
            'score': 0.0,
            'metric_name': 'MRR',
            'first_relevant_rank': None
        }
    
    @property
    def name(self) -> str:
        """Retorna el nombre de la métrica."""
        return "MRR"
