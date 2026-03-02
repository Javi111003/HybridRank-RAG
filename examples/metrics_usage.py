"""
Ejemplo de uso de las métricas de evaluación para sistemas RAG.

Este script demuestra cómo usar las diferentes métricas implementadas
para evaluar la calidad de un sistema de recuperación.
"""

import sys
from pathlib import Path

# Añadir el directorio raíz al path para importar src
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src.retriever.metrics import RecallAtK, PrecisionAtK, F1AtK, MRR, MAP, NDCG


def main():
    """Ejemplo de uso de las métricas."""
    
    # Simular documentos recuperados por el sistema (ordenados por score descendente)
    # Cada tupla es (document_id, relevance_score)
    retrieved_documents = [
        ("chunk_001", 0.95),
        ("chunk_042", 0.87),
        ("chunk_123", 0.76),
        ("chunk_089", 0.68),
        ("chunk_200", 0.55),
        ("chunk_150", 0.42),
    ]
    
    # Documentos que son realmente relevantes para la consulta
    relevant_documents = ["chunk_001", "chunk_123", "chunk_150", "chunk_999"]
    
    print("=" * 70)
    print("EVALUACIÓN DE SISTEMA DE RECUPERACIÓN")
    print("=" * 70)
    print(f"\nDocumentos recuperados: {len(retrieved_documents)}")
    print(f"Documentos relevantes totales: {len(relevant_documents)}")
    print(f"Relevantes recuperados: {len(set([d[0] for d in retrieved_documents]) & set(relevant_documents))}")
    print()
    
    # =========================================================================
    # Recall@k - ¿Cuántos documentos relevantes capturamos en top-k?
    # =========================================================================
    print("-" * 70)
    print("RECALL@k - Cobertura de documentos relevantes")
    print("-" * 70)
    
    recall_metric = RecallAtK()
    
    for k in [1, 3, 5]:
        result = recall_metric.compute(retrieved_documents, relevant_documents, k=k)
        print(f"\n{result['metric_name']}:")
        print(f"  Score: {result['score']:.3f}")
        print(f"  Documentos relevantes encontrados: {result['relevant_found']}/{result['total_relevant']}")
    
    # =========================================================================
    # Precision@k - ¿Cuántos documentos recuperados son realmente relevantes?
    # =========================================================================
    print("\n" + "-" * 70)
    print("PRECISION@k - Pureza/Limpieza del ranking")
    print("-" * 70)
    
    precision_metric = PrecisionAtK()
    
    for k in [1, 3, 5]:
        result = precision_metric.compute(retrieved_documents, relevant_documents, k=k)
        print(f"\n{result['metric_name']}:")
        print(f"  Score: {result['score']:.3f}")
        print(f"  Documentos relevantes de {result['total_retrieved']} recuperados: {result['relevant_found']}")
    
    # =========================================================================
    # F1@k - Balance entre Precision y Recall
    # =========================================================================
    print("\n" + "-" * 70)
    print("F1@k - Balance entre cobertura y limpieza")
    print("-" * 70)
    
    f1_metric = F1AtK()
    
    for k in [3, 5]:
        result = f1_metric.compute(retrieved_documents, relevant_documents, k=k)
        print(f"\n{result['metric_name']}:")
        print(f"  Score: {result['score']:.3f}")
        print(f"  Precision@{k}: {result['precision']:.3f}")
        print(f"  Recall@{k}: {result['recall']:.3f}")
    
    # =========================================================================
    # MRR - ¿Qué tan arriba está el primer documento relevante?
    # =========================================================================
    print("\n" + "-" * 70)
    print("MRR (Mean Reciprocal Rank) - Posición del primer relevante")
    print("-" * 70)
    
    mrr_metric = MRR()
    result = mrr_metric.compute(retrieved_documents, relevant_documents)
    
    print(f"\n{result['metric_name']}:")
    print(f"  Score: {result['score']:.3f}")
    print(f"  Primera posición relevante: {result['first_relevant_rank']}")
    
    # =========================================================================
    # MAP - Promedio de precisiones en posiciones relevantes
    # =========================================================================
    print("\n" + "-" * 70)
    print("MAP (Mean Average Precision) - Calidad del ranking")
    print("-" * 70)
    
    map_metric = MAP()
    result = map_metric.compute(retrieved_documents, relevant_documents)
    
    print(f"\n{result['metric_name']}:")
    print(f"  Score: {result['score']:.3f}")
    print(f"  Precisiones en posiciones relevantes: {[f'{p:.3f}' for p in result['precisions_at_relevant']]}")
    print(f"  Relevantes recuperados: {result['num_relevant_retrieved']}")
    
    # También podemos calcular MAP@k
    result_k3 = map_metric.compute(retrieved_documents, relevant_documents, k=3)
    print(f"\n{result_k3['metric_name']}:")
    print(f"  Score: {result_k3['score']:.3f}")
    
    # =========================================================================
    # nDCG@k - Calidad del ranking con descuento posicional
    # =========================================================================
    print("\n" + "-" * 70)
    print("nDCG@k - Ranking con descuento logarítmico")
    print("-" * 70)
    
    ndcg_metric = NDCG()
    
    for k in [3, 5, 10]:
        result = ndcg_metric.compute(retrieved_documents, relevant_documents, k=k)
        print(f"\n{result['metric_name']}:")
        print(f"  Score: {result['score']:.3f}")
        print(f"  DCG: {result['dcg']:.3f}")
        print(f"  IDCG: {result['idcg']:.3f}")
    
    # =========================================================================
    # Evaluación completa
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESUMEN DE EVALUACIÓN")
    print("=" * 70)
    
    metrics_summary = [
        ("Recall@3", RecallAtK().compute(retrieved_documents, relevant_documents, k=3)),
        ("Precision@3", PrecisionAtK().compute(retrieved_documents, relevant_documents, k=3)),
        ("F1@3", F1AtK().compute(retrieved_documents, relevant_documents, k=3)),
        ("Recall@5", RecallAtK().compute(retrieved_documents, relevant_documents, k=5)),
        ("Precision@5", PrecisionAtK().compute(retrieved_documents, relevant_documents, k=5)),
        ("F1@5", F1AtK().compute(retrieved_documents, relevant_documents, k=5)),
        ("MRR", MRR().compute(retrieved_documents, relevant_documents)),
        ("MAP", MAP().compute(retrieved_documents, relevant_documents)),
        ("nDCG@5", NDCG().compute(retrieved_documents, relevant_documents, k=5)),
    ]
    
    print("\nMétrica          | Score")
    print("-" * 70)
    for metric_name, result in metrics_summary:
        print(f"{metric_name:16} | {result['score']:.4f}")
    
    print("\n" + "=" * 70)
    print("INTERPRETACIÓN:")
    print("=" * 70)
    print("""
• Recall@k alto (>0.7): El sistema captura la mayoría de documentos relevantes
• Precision@k alto (>0.7): La mayoría de docs recuperados son relevantes (bajo ruido)
• F1@k alto (>0.7): Buen balance entre cobertura y limpieza
• MRR alto (>0.5): Documentos relevantes aparecen temprano en el ranking  
• MAP alto (>0.6): Buena precisión general a lo largo del ranking
• nDCG@k alto (>0.7): Ranking de alta calidad con relevantes bien posicionados

Para RAG:
- Recall@k asegura que el LLM tenga acceso a información relevante
- Precision@k alto reduce ruido que puede confundir al LLM
- F1@k ayuda a encontrar el k óptimo (balance cobertura-limpieza)
- MRR/nDCG altos reducen "distracción" del LLM con contenido irrelevante
- Valores bajos pueden aumentar hallucinations en la respuesta generada

Trade-offs:
- Aumentar k mejora Recall pero puede degradar Precision
- k óptimo maximiza F1 (depende de cada aplicación)
    """)


if __name__ == "__main__":
    main()
