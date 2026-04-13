"""
Script de exploración rápida de retrievers.

Ejecuta queries de prueba y compara resultados de BM25 vs Dense.

Uso:
    python scripts/test_retrievers.py
    python scripts/test_retrievers.py "tu query personalizada"
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.retriever.bm25_retriever import BM25Retriever
from src.retriever.dense_retriever import DenseRetriever


def format_result(chunk_id: str, score: float, rank: int, content: str = None) -> str:
    """Formatea un resultado para mostrar."""
    result = f"  {rank}. {chunk_id[:36]} (score: {score:.4f})"
    if content:
        preview = content[:100].replace('\n', ' ')
        result += f"\n     {preview}..."
    return result


def compare_retrievers(query: str, top_k: int = 5):
    """Compara BM25 y Dense para una query."""
    print(f"\n{'='*70}")
    print(f"Query: {query}")
    print('='*70)

    # Inicializar retrievers
    bm25 = BM25Retriever()
    dense = DenseRetriever()

    # Recuperar resultados
    bm25_results = bm25.retrieve(query, top_k=top_k)
    dense_results = dense.retrieve(query, top_k=top_k)

    # Recuperar contenido para preview
    all_chunk_ids = list(set(
        [cid for cid, _ in bm25_results] +
        [cid for cid, _ in dense_results]
    ))

    docs = {}
    if all_chunk_ids:
        result = dense._collection.get(
            ids=all_chunk_ids[:top_k * 2],  # limitar para no saturar
            include=['documents']
        )
        docs = dict(zip(result['ids'], result['documents']))

    # Resultados BM25
    print(f"\n📊 BM25 (sparse retrieval):")
    if not bm25_results:
        print("  (sin resultados)")
    for rank, (chunk_id, score) in enumerate(bm25_results, 1):
        print(format_result(chunk_id, score, rank, docs.get(chunk_id)))

    # Resultados Dense
    print(f"\n🔍 Dense E5 (dense retrieval):")
    if not dense_results:
        print("  (sin resultados)")
    for rank, (chunk_id, score) in enumerate(dense_results, 1):
        print(format_result(chunk_id, score, rank, docs.get(chunk_id)))

    # Overlap
    bm25_ids = set(cid for cid, _ in bm25_results)
    dense_ids = set(cid for cid, _ in dense_results)
    overlap = bm25_ids & dense_ids

    print(f"\n🔀 Overlap: {len(overlap)}/{top_k} documentos en común")
    if overlap:
        print(f"   Chunk IDs: {', '.join([cid[:8] for cid in list(overlap)[:3]]}...")


def main():
    # Queries de ejemplo
    test_queries = [
        "educación superior universidades",
        "decreto ley inversión extranjera",
        "derechos trabajadores salario mínimo",
        "procedimiento divorcio judicial",
    ]

    # Si se proporciona una query como argumento, usarla
    if len(sys.argv) > 1:
        custom_query = ' '.join(sys.argv[1:])
        compare_retrievers(custom_query, top_k=5)
    else:
        # Ejecutar queries de ejemplo
        print("\n🚀 Probando retrievers con queries de ejemplo...")
        for query in test_queries:
            compare_retrievers(query, top_k=5)
            print()

        print("\n💡 Tip: Ejecuta con tu propia query:")
        print("    python scripts/test_retrievers.py \"tu query aquí\"")


if __name__ == '__main__':
    main()
