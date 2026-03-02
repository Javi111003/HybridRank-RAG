# Abstract class for evaluation metrics
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional

class Metric(ABC):
    """
    Clase base abstracta para métricas de evaluación de sistemas de recuperación.
    """
    
    @abstractmethod
    def compute(
        self,
        retrieved_documents: List[Tuple[str, float]],
        relevant_documents: List[str],
        k: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calcula el valor de la métrica.
        
        Args:
            retrieved_documents: Lista de tuplas (doc_id, score) ordenadas por relevancia
                                descendente. doc_id es el identificador único del documento
                                (chunk_id) y score es la puntuación de relevancia.
            relevant_documents: Lista de doc_ids que son relevantes para la consulta.
            k: Número de documentos top a considerar. Si es None, considera todos
               los documentos recuperados.
            **kwargs: Parámetros adicionales específicos de cada métrica.
        
        Returns:
            Dict con al menos:
                - 'score': float - Valor numérico de la métrica
                - 'metric_name': str - Nombre descriptivo de la métrica
                Puede incluir metadata adicional específica de cada métrica.
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Retorna el nombre de la métrica."""
        pass