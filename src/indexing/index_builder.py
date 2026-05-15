import json
import os
import sys
import pickle
import shutil
import logging
from typing import List, Dict, Any, Tuple, Iterator
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import ijson
import chromadb
from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, '.data')
EMBEDDINGS_PATH = os.path.join(DATA_DIR, 'embeddings', 'e5_elements_with_embeddings.json')
NORMA_EMBEDDINGS_PATH = os.path.join(DATA_DIR, 'embeddings', 'e5_norma_fragments_with_embeddings.json')
BM25_INDEX_DIR = os.path.join(DATA_DIR, 'bm25_index')
NORMA_BM25_INDEX_DIR = os.path.join(DATA_DIR, 'bm25_norma_index')
CHROMA_DIR = os.path.join(DATA_DIR, 'chroma')
NORMA_CHROMA_DIR = os.path.join(DATA_DIR, 'chroma_normas')
CHROMA_COLLECTION_NAME = "hybridrank_elements"
NORMA_CHROMA_COLLECTION_NAME = "hybridrank_normas"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"

BATCH_SIZE = 5000
TOKENIZE_BATCH_SIZE = 1000
TOKENIZE_N_PROCESS = 1


def _create_spanish_tokenizer():
    from src.retriever.tokenizer import SpanishTokenizer
    return SpanishTokenizer()


def _is_valid_element(elem: dict, require_embedding: bool = True) -> bool:
    """Verifica que un elemento tenga los campos requeridos para indexacion."""
    chunk_id = elem.get('metadata', {}).get('chunk_id')
    cleaned = elem.get('cleaned_content', '').strip()
    if not chunk_id or not cleaned:
        return False
    if require_embedding:
        embedding = elem.get('embedding', [])
        return bool(embedding)
    return True


def stream_elements(path: str, require_embedding: bool = True) -> Iterator[Dict[str, Any]]:
    """
    Genera elementos validos uno a uno desde el JSON usando ijson (streaming).
    No carga el archivo completo en memoria — ideal para archivos de varios GB.
    """
    logger.info(f"Streaming elementos desde {path}...")
    with open(path, 'rb') as f:
        for elem in ijson.items(f, 'item'):
            if _is_valid_element(elem, require_embedding=require_embedding):
                yield elem


def stream_batches(
    path: str,
    batch_size: int = BATCH_SIZE,
    require_embedding: bool = True,
) -> Iterator[List[Dict[str, Any]]]:
    """
    Genera batches de elementos validos desde el JSON usando streaming.
    Cada batch es una lista de hasta batch_size elementos.
    """
    batch = []
    for elem in stream_elements(path, require_embedding=require_embedding):
        batch.append(elem)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_bm25_index(
    path: str,
    index_dir: str,
    batch_size: int = BATCH_SIZE,
    tokenize_batch_size: int = TOKENIZE_BATCH_SIZE,
    n_process: int = TOKENIZE_N_PROCESS,
) -> None:
    """
    Construye un indice BM25 desde el archivo JSON usando streaming + batch tokenization.

    Lee el archivo por lotes con ijson, tokeniza cada lote con spaCy nlp.pipe(),
    y acumula solo los tokens (livianos) y doc_ids en memoria.
    Los embeddings (768 floats por elemento) nunca se acumulan.

    :param path: Ruta al archivo JSON de embeddings.
    :param index_dir: Directorio donde guardar el indice.
    :param batch_size: Elementos por lote de lectura.
    :param tokenize_batch_size: Tamaño del batch interno de spaCy pipe.
    :param n_process: Numero de procesos para tokenizacion paralela.
    """
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
    os.makedirs(index_dir, exist_ok=True)

    tokenizer = _create_spanish_tokenizer()
    all_doc_ids = []
    all_tokenized = []
    total_read = 0
    empty_count = 0

    logger.info(f"Tokenizando corpus en batches de {batch_size} (spaCy batch={tokenize_batch_size}, "
                f"n_process={n_process})...")

    for batch in stream_batches(path, batch_size, require_embedding=False):
        contents = [elem['cleaned_content'] for elem in batch]
        chunk_ids = [elem['metadata']['chunk_id'] for elem in batch]

        tokenized = tokenizer.tokenize_batch(
            contents,
            batch_size=tokenize_batch_size,
            n_process=n_process,
        )

        for chunk_id, tokens in zip(chunk_ids, tokenized):
            if tokens:
                all_doc_ids.append(chunk_id)
                all_tokenized.append(tokens)
            else:
                empty_count += 1

        total_read += len(batch)
        logger.info(f"BM25 tokenizacion: {total_read} elementos procesados...")

    if empty_count > 0:
        logger.warning(f"{empty_count} documentos con tokenizacion vacia omitidos.")
    logger.info(f"Corpus tokenizado: {len(all_tokenized)} documentos de {total_read} leidos.")

    logger.info("Construyendo modelo BM25Okapi...")
    bm25 = BM25Okapi(all_tokenized)
    logger.info(f"Modelo BM25 construido: corpus_size={bm25.corpus_size}, avgdl={bm25.avgdl:.2f}")

    model_path = os.path.join(index_dir, 'bm25_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(bm25, f)
    logger.info(f"Modelo BM25 guardado en {model_path}")

    doc_ids_path = os.path.join(index_dir, 'doc_ids.json')
    with open(doc_ids_path, 'w', encoding='utf-8') as f:
        json.dump(all_doc_ids, f, ensure_ascii=False)
    logger.info(f"Mapeo de IDs guardado: {len(all_doc_ids)} documentos en {doc_ids_path}")

    metadata_path = os.path.join(index_dir, 'metadata.json')
    metadata = {
        "num_docs": len(all_doc_ids),
        "avg_doc_length": float(bm25.avgdl),
        "k1": bm25.k1,
        "b": bm25.b,
        "tokenizer": "es_core_news_md",
        "embedding_model": EMBEDDING_MODEL_NAME,
        "tokenize_batch_size": tokenize_batch_size,
        "n_process": n_process,
        "created_at": datetime.now().isoformat()
    }
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info("Indice BM25 construido exitosamente.")


_METADATA_KEYS_TO_KEEP = {
    'chunk_id',
    'source',
    'type',
    'filename',
    'document_type',
    'filetype',
    'page_number',
    'corpus_type',
    'tipo',
    'numero',
    'year',
    'organismo_emisor',
    'goc_code',
    'page_start',
    'page_end',
    'match_confidence',
    'ordinal_position',
    'raw_metadata_string',
}

_METADATA_PREFIXES_TO_KEEP = ('gaceta_', 'norma_', 'fragment_')


def _chroma_safe_metadata_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _sanitize_metadata(meta: dict) -> dict:
    """
    Asegura que todos los valores de metadata sean compatibles con ChromaDB.
    ChromaDB solo acepta str, int, float, bool como valores.
    Preserva campos gaceta_*, norma_* y fragment_* para permitir filtrado
    y trazabilidad en retrieval.
    """
    clean = {}
    for key in ['source', 'type', 'filename', 'document_type', 'filetype']:
        clean[key] = _chroma_safe_metadata_value(meta.get(key) or "")
    clean['page_number'] = meta.get('page_number') if meta.get('page_number') is not None else -1

    for key, value in meta.items():
        if key not in _METADATA_KEYS_TO_KEEP and not key.startswith(_METADATA_PREFIXES_TO_KEEP):
            continue
        clean[key] = _chroma_safe_metadata_value(value)

    return clean


def _prepare_chroma_batch(elements: List[Dict[str, Any]]) -> Tuple[list, list, list, list]:
    """Prepara un batch de elementos para insercion en ChromaDB."""
    ids = []
    embeddings = []
    documents = []
    metadatas = []
    for elem in elements:
        ids.append(elem['metadata']['chunk_id'])
        embeddings.append([float(x) for x in elem['embedding']])
        documents.append(elem.get('content', elem['cleaned_content']))
        metadatas.append(_sanitize_metadata(elem['metadata']))
    return ids, embeddings, documents, metadatas


def build_chroma_index(
    path: str,
    chroma_dir: str,
    collection_name: str,
    batch_size: int = BATCH_SIZE,
) -> None:
    """
    Carga embeddings y metadata en ChromaDB usando streaming por lotes.

    Lee el archivo JSON con ijson en batches, prepara y inserta cada batch
    sin acumular el dataset completo en memoria. Usa un ThreadPoolExecutor
    para preparar el siguiente batch mientras se inserta el actual.
    """
    client = chromadb.PersistentClient(path=chroma_dir)

    try:
        client.delete_collection(name=collection_name)
        logger.info(f"Coleccion '{collection_name}' existente eliminada.")
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    logger.info(f"Coleccion '{collection_name}' creada con distancia coseno.")

    total_inserted = 0
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending_future = None

        for batch in stream_batches(path, batch_size):
            future = executor.submit(_prepare_chroma_batch, batch)

            if pending_future is not None:
                p_ids, p_embs, p_docs, p_metas = pending_future.result()
                collection.add(ids=p_ids, embeddings=p_embs, documents=p_docs, metadatas=p_metas)
                total_inserted += len(p_ids)
                logger.info(f"ChromaDB: insertados {total_inserted} elementos.")

            pending_future = future

        if pending_future is not None:
            p_ids, p_embs, p_docs, p_metas = pending_future.result()
            collection.add(ids=p_ids, embeddings=p_embs, documents=p_docs, metadatas=p_metas)
            total_inserted += len(p_ids)
            logger.info(f"ChromaDB: insertados {total_inserted} elementos.")

    logger.info(f"Verificacion ChromaDB: {collection.count()} documentos en coleccion.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Construye indices BM25 y/o Chroma para HybridRank")
    parser.add_argument(
        '--input-file',
        default=EMBEDDINGS_PATH,
        help="JSON con elementos y embeddings. Default: .data/embeddings/e5_elements_with_embeddings.json",
    )
    parser.add_argument(
        '--build',
        choices=['bm25', 'chroma', 'both'],
        default='chroma',
        help="Indice a construir (default: chroma, para preservar el comportamiento actual)",
    )
    parser.add_argument(
        '--bm25-index-dir',
        default=BM25_INDEX_DIR,
        help="Directorio de salida para BM25",
    )
    parser.add_argument(
        '--chroma-dir',
        default=CHROMA_DIR,
        help="Directorio persistente de ChromaDB",
    )
    parser.add_argument(
        '--collection-name',
        default=CHROMA_COLLECTION_NAME,
        help="Nombre de la coleccion ChromaDB",
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=BATCH_SIZE,
        help=f"Tamano de batch para lectura/insercion (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        '--tokenize-batch-size',
        type=int,
        default=TOKENIZE_BATCH_SIZE,
        help=f"Tamano de batch de spaCy para BM25 (default: {TOKENIZE_BATCH_SIZE})",
    )
    parser.add_argument(
        '--n-process',
        type=int,
        default=TOKENIZE_N_PROCESS,
        help=f"Procesos spaCy para BM25 (default: {TOKENIZE_N_PROCESS})",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Inicio de indexacion HybridRank")
    logger.info("=" * 60)

    if not os.path.exists(args.input_file):
        logger.error(f"No se encontro el archivo de entrada: {args.input_file}")
        sys.exit(1)

    if args.build in ('bm25', 'both'):
        logger.info("\n--- Construyendo indice BM25 (rank-bm25) ---")
        build_bm25_index(
            args.input_file,
            args.bm25_index_dir,
            batch_size=args.batch_size,
            tokenize_batch_size=args.tokenize_batch_size,
            n_process=args.n_process,
        )

    if args.build in ('chroma', 'both'):
        logger.info("\n--- Construyendo indice ChromaDB (Dense) ---")
        build_chroma_index(
            args.input_file,
            args.chroma_dir,
            args.collection_name,
            batch_size=args.batch_size,
        )

    logger.info("\n" + "=" * 60)
    logger.info("Indexacion completada exitosamente.")
    logger.info(f"  Input:       {args.input_file}")
    logger.info(f"  Build:       {args.build}")
    logger.info(f"  BM25 index:  {args.bm25_index_dir}")
    logger.info(f"  ChromaDB:    {args.chroma_dir}")
    logger.info(f"  Collection:  {args.collection_name}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
