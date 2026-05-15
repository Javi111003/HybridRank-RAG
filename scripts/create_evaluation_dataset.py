"""
Script interactivo de pooling para construir un dataset de evaluacion de retrieval.

Para cada query, ejecuta BM25Retriever y DenseRetriever, une los resultados,
muestra un pool con previews del contenido, y permite anotar relevancia manual
via input(). Guarda el resultado en JSON.

Uso:
    python scripts/create_evaluation_dataset.py --queries scripts/sample_queries.json
    python scripts/create_evaluation_dataset.py --queries queries.json --output results.json --top-k 15
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.retriever.bm25_retriever import BM25Retriever
from src.retriever.dense_retriever import DenseRetriever

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

CORPUS_CONFIGS = {
    "elements": {
        "output": os.path.join(PROJECT_ROOT, ".data", "evaluation", "qrels.json"),
        "bm25_index_dir": os.path.join(PROJECT_ROOT, ".data", "bm25_index"),
        "chroma_dir": os.path.join(PROJECT_ROOT, ".data", "chroma"),
        "collection_name": "hybridrank_elements",
    },
    "normas": {
        "output": os.path.join(PROJECT_ROOT, ".data", "evaluation", "norma_qrels.json"),
        "bm25_index_dir": os.path.join(PROJECT_ROOT, ".data", "bm25_norma_index"),
        "chroma_dir": os.path.join(PROJECT_ROOT, ".data", "chroma_normas"),
        "collection_name": "hybridrank_normas",
    },
}


# ---------------------------------------------------------------------------
# Tipos
# ---------------------------------------------------------------------------

# Cada entrada del pool es un dict con:
#   doc_id: str
#   sources: List[str]       -- nombres de retrievers que lo devolvieron
#   score_bm25: float | None
#   score_dense: float | None
PoolEntry = Dict


# ---------------------------------------------------------------------------
# Funciones modulares
# ---------------------------------------------------------------------------


def load_queries(path: str) -> List[Dict]:
    """Carga queries desde un JSON con estructura [{query_id, query_type, query}]."""
    with open(path, "r", encoding="utf-8") as f:
        queries = json.load(f)
    for q in queries:
        if not all(k in q for k in ("query_id", "query_type", "query")):
            raise ValueError(
                f"Cada query debe tener 'query_id', 'query_type' y 'query'. "
                f"Encontrado: {list(q.keys())}"
            )
    return queries


def resolve_corpus_config(
    corpus: str,
    output: Optional[str] = None,
    bm25_index_dir: Optional[str] = None,
    chroma_dir: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> Dict[str, str]:
    """Resuelve rutas para el corpus seleccionado."""
    if corpus not in CORPUS_CONFIGS:
        raise ValueError(f"Corpus no soportado: {corpus}")

    defaults = CORPUS_CONFIGS[corpus]
    return {
        "output": output or defaults["output"],
        "bm25_index_dir": bm25_index_dir or defaults["bm25_index_dir"],
        "chroma_dir": chroma_dir or defaults["chroma_dir"],
        "collection_name": collection_name or defaults["collection_name"],
    }


def create_retrievers(config: Dict[str, str]) -> Tuple[BM25Retriever, DenseRetriever]:
    """Crea retrievers usando rutas explicitas del corpus."""
    bm25 = BM25Retriever(index_dir=config["bm25_index_dir"])
    dense = DenseRetriever(
        chroma_dir=config["chroma_dir"],
        collection_name=config["collection_name"],
    )
    return bm25, dense


def build_pool(
    query: str,
    bm25: BM25Retriever,
    dense: DenseRetriever,
    top_k: int = 10,
) -> List[PoolEntry]:
    """
    Construye un pool de documentos candidatos uniendo resultados de ambos retrievers.

    Retorna lista de PoolEntry ordenada: primero los encontrados por ambos,
    luego por el mayor score disponible.
    """
    bm25_results = bm25.retrieve(query, top_k=top_k)
    dense_results = dense.retrieve(query, top_k=top_k)

    # Indexar por doc_id
    pool_map: Dict[str, PoolEntry] = {}

    for doc_id, score in bm25_results:
        pool_map[doc_id] = {
            "doc_id": doc_id,
            "sources": ["BM25"],
            "score_bm25": float(score),
            "score_dense": None,
        }

    for doc_id, score in dense_results:
        if doc_id in pool_map:
            pool_map[doc_id]["sources"].append("Dense")
            pool_map[doc_id]["score_dense"] = float(score)
        else:
            pool_map[doc_id] = {
                "doc_id": doc_id,
                "sources": ["Dense"],
                "score_bm25": None,
                "score_dense": float(score),
            }

    # Ordenar: ambos primero, luego por max score descendente
    def sort_key(entry: PoolEntry) -> Tuple[int, float]:
        both = 1 if len(entry["sources"]) > 1 else 0
        max_score = max(
            s for s in (entry["score_bm25"], entry["score_dense"]) if s is not None
        )
        return (both, max_score)

    return sorted(pool_map.values(), key=sort_key, reverse=True)


def fetch_previews(
    doc_ids: List[str], chroma_collection
) -> Dict[str, str]:
    """Recupera un preview (primeros 200 chars) de cada chunk desde ChromaDB."""
    if not doc_ids:
        return {}

    result = chroma_collection.get(ids=doc_ids, include=["documents"])
    previews = {}
    for chunk_id, doc in zip(result["ids"], result["documents"]):
        text = doc or ""
        previews[chunk_id] = text[:200] + ("..." if len(text) > 200 else "")
    return previews


def _clean_whitespace(text: str) -> str:
    """Elimina saltos de linea y espacios en blanco innecesarios para legibilidad."""
    import re
    # Colapsar multiples saltos de linea en uno solo
    text = re.sub(r'\n\s*\n+', '\n', text)
    # Colapsar espacios multiples (sin tocar saltos de linea)
    text = re.sub(r'[^\S\n]+', ' ', text)
    # Limpiar espacios al inicio/final de cada linea
    lines = [line.strip() for line in text.split('\n')]
    return '\n'.join(line for line in lines if line).strip()


def fetch_full_texts(
    doc_ids: List[str], chroma_collection
) -> Dict[str, str]:
    """Recupera el texto completo de cada chunk desde ChromaDB, con limpieza de espacios."""
    if not doc_ids:
        return {}

    result = chroma_collection.get(ids=doc_ids, include=["documents"])
    texts = {}
    for chunk_id, doc in zip(result["ids"], result["documents"]):
        text = _clean_whitespace(doc or "")
        texts[chunk_id] = text
    return texts


def fetch_doc_metadata(
    doc_ids: List[str], chroma_collection
) -> Dict[str, Dict[str, Any]]:
    """Recupera metadata de ChromaDB para trazabilidad del pool."""
    if not doc_ids:
        return {}

    result = chroma_collection.get(ids=doc_ids, include=["metadatas"])
    metadata_by_id: Dict[str, Dict[str, Any]] = {}
    for doc_id, metadata in zip(result["ids"], result["metadatas"]):
        metadata_by_id[doc_id] = metadata or {}
    return metadata_by_id


def _unique_norma_ids(doc_ids: List[str], doc_metadata: Dict[str, Dict[str, Any]]) -> List[str]:
    seen = set()
    norma_ids = []
    for doc_id in doc_ids:
        norma_id = doc_metadata.get(doc_id, {}).get("norma_id")
        if not norma_id or norma_id in seen:
            continue
        seen.add(norma_id)
        norma_ids.append(norma_id)
    return norma_ids


def build_result_record(
    query_info: Dict,
    doc_ids: List[str],
    relevant_docs: List[str],
    corpus: str,
    doc_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict:
    """Construye el registro qrels manteniendo compatibilidad con el formato actual."""
    record = {
        "query_id": query_info["query_id"],
        "query_type": query_info["query_type"],
        "query": query_info["query"],
        "pool_docs": doc_ids,
        "relevant_docs": relevant_docs,
    }

    if corpus == "normas":
        metadata = doc_metadata or {}
        record["pool_normas"] = _unique_norma_ids(doc_ids, metadata)
        record["relevant_normas"] = _unique_norma_ids(relevant_docs, metadata)
        record["doc_metadata"] = {
            doc_id: metadata.get(doc_id, {})
            for doc_id in doc_ids
        }

    return record


def display_pool(
    query_info: Dict,
    pool: List[PoolEntry],
    previews: Dict[str, str],
) -> None:
    """Muestra el pool de documentos candidatos en consola de forma legible."""
    sep = "=" * 90
    print(f"\n{sep}")
    print(f"  Query ID:   {query_info['query_id']}")
    print(f"  Tipo:       {query_info['query_type']}")
    print(f"  Query:      {query_info['query']}")
    print(f"  Pool size:  {len(pool)} documentos")
    print(sep)

    header = f"{'#':>3}  {'Doc ID':<30}  {'Origen':<10}  {'BM25':>8}  {'Dense':>8}  Preview"
    print(header)
    print("-" * 90)

    for idx, entry in enumerate(pool, 1):
        origen = "Ambos" if len(entry["sources"]) > 1 else entry["sources"][0]
        bm25_str = f"{entry['score_bm25']:.4f}" if entry["score_bm25"] is not None else "   -"
        dense_str = f"{entry['score_dense']:.4f}" if entry["score_dense"] is not None else "   -"
        preview = previews.get(entry["doc_id"], "")
        # Truncar preview para que quepa en una linea
        preview_short = preview[:60] + "..." if len(preview) > 60 else preview
        print(f"{idx:>3}  {entry['doc_id']:<30}  {origen:<10}  {bm25_str:>8}  {dense_str:>8}  {preview_short}")

    print("-" * 90)


def annotate_relevant(pool: List[PoolEntry]) -> List[str]:
    """
    Pide al usuario que marque los documentos relevantes via input().

    El usuario introduce indices separados por coma (ej: 1,3,5) o 'none'
    si ninguno es relevante. Retorna lista de doc_ids relevantes.
    """
    while True:
        raw = input(
            "\n  Indices de docs relevantes (ej: 1,3,5) o 'none' si ninguno: "
        ).strip()

        if raw.lower() == "none":
            return []

        try:
            indices = [int(x.strip()) for x in raw.split(",") if x.strip()]
        except ValueError:
            print("  Error: introduce numeros separados por coma, o 'none'.")
            continue

        # Validar rango
        invalid = [i for i in indices if i < 1 or i > len(pool)]
        if invalid:
            print(f"  Error: indices fuera de rango: {invalid}. Rango valido: 1-{len(pool)}")
            continue

        return [pool[i - 1]["doc_id"] for i in indices]


# ---------------------------------------------------------------------------
# Modo paginado (--paginated)
# ---------------------------------------------------------------------------

# Estados de anotacion por documento
_UNSET = 0
_RELEVANT = 1
_NOT_RELEVANT = 2

_STATUS_LABELS = {
    _UNSET: "[ ] Sin anotar",
    _RELEVANT: "[✓] Relevante",
    _NOT_RELEVANT: "[✗] No relevante",
}


def display_single_doc(
    idx: int,
    total: int,
    entry: PoolEntry,
    full_text: str,
    status: int,
    query_text: str,
) -> None:
    """Renderiza un documento individual en modo paginado."""
    sep = "─" * 70
    print(f"\n{sep}")
    print(f"  Doc {idx + 1}/{total}")
    print(sep)
    print(f"  ID:      {entry['doc_id']}")
    origen = "Ambos" if len(entry["sources"]) > 1 else entry["sources"][0]
    bm25_str = f"{entry['score_bm25']:.4f}" if entry["score_bm25"] is not None else "-"
    dense_str = f"{entry['score_dense']:.4f}" if entry["score_dense"] is not None else "-"
    print(f"  Origen:  {origen}    BM25: {bm25_str}    Dense: {dense_str}")
    print(f"  Estado:  {_STATUS_LABELS[status]}")
    print(f"\n  Query: {query_text}")
    print(f"\n  Contenido:")
    # Word-wrap manual a ~80 cols con indentacion
    for line in _wrap_text(full_text, width=76):
        print(f"    {line}")
    print(f"\n  [y] Relevante  [n] No relevante  [b] Atras  [s] Saltar  [d] Done  [l] Lista  [?] Ayuda")
    print(sep)


def _wrap_text(text: str, width: int = 76) -> List[str]:
    """Divide texto en lineas de hasta width caracteres respetando saltos existentes."""
    lines = []
    for paragraph in text.split("\n"):
        while len(paragraph) > width:
            # Buscar ultimo espacio antes del limite
            break_at = paragraph.rfind(" ", 0, width)
            if break_at == -1:
                break_at = width
            lines.append(paragraph[:break_at])
            paragraph = paragraph[break_at:].lstrip()
        lines.append(paragraph)
    return lines


def _display_status_list(pool: List[PoolEntry], statuses: List[int]) -> None:
    """Muestra resumen compacto del estado de anotacion de todos los docs."""
    print("\n  Estado de anotacion:")
    for i, (entry, status) in enumerate(zip(pool, statuses)):
        marker = _STATUS_LABELS[status]
        print(f"    {i + 1:>3}. {entry['doc_id']:<30}  {marker}")
    print()


def _display_help() -> None:
    """Muestra ayuda de comandos del modo paginado."""
    print("""
  Comandos:
    y  - Marcar como relevante y avanzar
    n  - Marcar como no relevante y avanzar
    b  - Retroceder al documento anterior
    s  - Saltar sin anotar y avanzar
    d  - Finalizar anotacion (pide confirmacion)
    l  - Ver lista de estado de todos los documentos
    ?  - Mostrar esta ayuda
""")


def annotate_paginated(
    pool: List[PoolEntry],
    full_texts: Dict[str, str],
    query_info: Dict,
) -> List[str]:
    """
    Modo paginado: navega doc por doc, marca relevancia binaria (y/n),
    permite retroceder y avanzar. Retorna lista de doc_ids relevantes.
    """
    total = len(pool)
    statuses = [_UNSET] * total
    cursor = 0

    while True:
        entry = pool[cursor]
        text = full_texts.get(entry["doc_id"], "(sin contenido)")
        display_single_doc(cursor, total, entry, text, statuses[cursor], query_info["query"])

        raw = input("  > ").strip().lower()

        if raw == "y":
            statuses[cursor] = _RELEVANT
            if cursor < total - 1:
                cursor += 1
            else:
                print("  (ultimo documento — usa 'd' para finalizar)")

        elif raw == "n":
            statuses[cursor] = _NOT_RELEVANT
            if cursor < total - 1:
                cursor += 1
            else:
                print("  (ultimo documento — usa 'd' para finalizar)")

        elif raw == "b":
            if cursor > 0:
                cursor -= 1
            else:
                print("  (ya estas en el primer documento)")

        elif raw == "s":
            if cursor < total - 1:
                cursor += 1
            else:
                print("  (ultimo documento — usa 'd' para finalizar)")

        elif raw == "l":
            _display_status_list(pool, statuses)

        elif raw == "?":
            _display_help()

        elif raw == "d":
            # Resumen antes de confirmar
            n_relevant = statuses.count(_RELEVANT)
            n_not_relevant = statuses.count(_NOT_RELEVANT)
            n_unset = statuses.count(_UNSET)
            print(f"\n  Resumen: {n_relevant} relevantes, {n_not_relevant} no relevantes, {n_unset} sin anotar.")

            if n_unset > 0:
                confirm = input(f"  Hay {n_unset} docs sin anotar. ¿Finalizar de todas formas? [y/n]: ").strip().lower()
                if confirm != "y":
                    continue
            else:
                confirm = input("  ¿Confirmar anotacion? [y/n]: ").strip().lower()
                if confirm != "y":
                    continue

            return [
                pool[i]["doc_id"]
                for i in range(total)
                if statuses[i] == _RELEVANT
            ]

        else:
            print("  Comando no reconocido. Escribe '?' para ayuda.")


def save_results(results: List[Dict], output_path: str) -> None:
    """Guarda los resultados de anotacion en JSON."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResultados guardados en: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Pooling interactivo para dataset de evaluacion de retrieval"
    )
    parser.add_argument(
        "--queries", required=True, help="Path al JSON de queries"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path de salida del JSON (default: qrels por corpus)",
    )
    parser.add_argument(
        "--corpus",
        choices=["elements", "normas"],
        default="elements",
        help="Corpus a evaluar: elements usa el indice actual, normas usa fragmentos normativos",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Cantidad de documentos a recuperar por retriever (default: 10)",
    )
    parser.add_argument(
        "--paginated",
        action="store_true",
        help="Modo paginado: navega doc por doc con contenido extendido y anotacion binaria",
    )
    parser.add_argument(
        "--bm25-index-dir",
        default=None,
        help="Directorio BM25 explicito; si se omite, depende del corpus",
    )
    parser.add_argument(
        "--chroma-dir",
        default=None,
        help="Directorio ChromaDB explicito; si se omite, depende del corpus",
    )
    parser.add_argument(
        "--collection-name",
        default=None,
        help="Coleccion ChromaDB explicita; si se omite, depende del corpus",
    )
    args = parser.parse_args()
    config = resolve_corpus_config(
        args.corpus,
        output=args.output,
        bm25_index_dir=args.bm25_index_dir,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection_name,
    )

    queries = load_queries(args.queries)
    print(f"Cargadas {len(queries)} queries desde {args.queries}")

    print(f"Inicializando retrievers para corpus '{args.corpus}'...")
    bm25, dense = create_retrievers(config)
    chroma_collection = dense._collection
    print("Retrievers listos.\n")

    results = []

    for i, query_info in enumerate(queries):
        print(f"\n[{i + 1}/{len(queries)}]")

        # Construir pool
        pool = build_pool(query_info["query"], bm25, dense, top_k=args.top_k)

        # Obtener previews del contenido
        doc_ids = [e["doc_id"] for e in pool]
        previews = fetch_previews(doc_ids, chroma_collection)
        doc_metadata = (
            fetch_doc_metadata(doc_ids, chroma_collection)
            if args.corpus == "normas"
            else {}
        )

        # Mostrar pool resumen (ambos modos)
        display_pool(query_info, pool, previews)

        # Anotar relevancia
        if args.paginated:
            full_texts = fetch_full_texts(doc_ids, chroma_collection)
            relevant_docs = annotate_paginated(pool, full_texts, query_info)
        else:
            relevant_docs = annotate_relevant(pool)

        results.append(build_result_record(
            query_info,
            doc_ids,
            relevant_docs,
            corpus=args.corpus,
            doc_metadata=doc_metadata,
        ))

        print(f"  -> {len(relevant_docs)} docs marcados como relevantes.")

    # Guardar
    save_results(results, config["output"])

    # Resumen
    total_relevant = sum(len(r["relevant_docs"]) for r in results)
    total_pool = sum(len(r["pool_docs"]) for r in results)
    print(f"\nResumen: {total_relevant}/{total_pool} documentos marcados relevantes en {len(results)} queries.")


if __name__ == "__main__":
    main()
