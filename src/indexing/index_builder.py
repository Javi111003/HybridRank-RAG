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

from src.retriever.tokenizer import SpanishTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, '.data')
EMBEDDINGS_PATH = os.path.join(DATA_DIR, 'embeddings', 'e5_elements_with_embeddings.json')
BM25_INDEX_DIR = os.path.join(DATA_DIR, 'bm25_index')
CHROMA_DIR = os.path.join(DATA_DIR, 'chroma')
CHROMA_COLLECTION_NAME = "hybridrank_elements"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"

BATCH_SIZE = 5000
TOKENIZE_BATCH_SIZE = 1000
TOKENIZE_N_PROCESS = 1


def _is_valid_element(elem: dict) -> bool:
    """Verifica que un elemento tenga los campos requeridos para indexacion."""
    chunk_id = elem.get('metadata', {}).get('chunk_id')
    cleaned = elem.get('cleaned_content', '').strip()
    embedding = elem.get('embedding', [])
    return bool(chunk_id and cleaned and embedding)


def stream_elements(path: str) -> Iterator[Dict[str, Any]]:
    """
    Genera elementos validos uno a uno desde el JSON usando ijson (streaming).
    No carga el archivo completo en memoria — ideal para archivos de varios GB.
    """
    logger.info(f"Streaming elementos desde {path}...")
    with open(path, 'rb') as f:
        for elem in ijson.items(f, 'item'):
            if _is_valid_element(elem):
                yield elem


def stream_batches(path: str, batch_size: int = BATCH_SIZE) -> Iterator[List[Dict[str, Any]]]:
    """
    Genera batches de elementos validos desde el JSON usando streaming.
    Cada batch es una lista de hasta batch_size elementos.
    """
    batch = []
    for elem in stream_elements(path):
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

    tokenizer = SpanishTokenizer()
    all_doc_ids = []
    all_tokenized = []
    total_read = 0
    empty_count = 0

    logger.info(f"Tokenizando corpus en batches de {batch_size} (spaCy batch={tokenize_batch_size}, "
                f"n_process={n_process})...")

    for batch in stream_batches(path, batch_size):
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


def _sanitize_metadata(meta: dict) -> dict:
    """
    Asegura que todos los valores de metadata sean compatibles con ChromaDB.
    ChromaDB solo acepta str, int, float, bool como valores.
    Preserva campos gaceta_* para permitir filtrado en retrieval.
    """
    clean = {}
    for key in ['source', 'type', 'filename', 'document_type', 'filetype']:
        clean[key] = meta.get(key) or ""
    clean['page_number'] = meta.get('page_number') if meta.get('page_number') is not None else -1

    for key, value in meta.items():
        if not key.startswith('gaceta_'):
            continue
        if value is None:
            clean[key] = ""
        elif isinstance(value, list):
            clean[key] = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = str(value)

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
    logger.info("=" * 60)
    logger.info("Inicio de indexacion HybridRank")
    logger.info("=" * 60)

    if not os.path.exists(EMBEDDINGS_PATH):
        logger.error(f"No se encontro el archivo de embeddings: {EMBEDDINGS_PATH}")
        sys.exit(1)

    #logger.info("\n--- Paso 1/2: Construyendo indice BM25 (rank-bm25) ---")
    #build_bm25_index(EMBEDDINGS_PATH, BM25_INDEX_DIR)

    logger.info("\n--- Paso 2/2: Construyendo indice ChromaDB (Dense) ---")
    build_chroma_index(EMBEDDINGS_PATH, CHROMA_DIR, CHROMA_COLLECTION_NAME)

    logger.info("\n" + "=" * 60)
    logger.info("Indexacion completada exitosamente.")
    logger.info(f"  BM25 index:  {BM25_INDEX_DIR}")
    logger.info(f"  ChromaDB:    {CHROMA_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
