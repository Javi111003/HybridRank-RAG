import json
import os
import sys
import pickle
import shutil
import logging
from typing import List, Dict, Any
from datetime import datetime

import chromadb
from rank_bm25 import BM25Okapi

from src.retriever.tokenizer import SpanishTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, '.data')
EMBEDDINGS_PATH = os.path.join(DATA_DIR, 'embeddings', 'elements_with_embeddings.json')
BM25_INDEX_DIR = os.path.join(DATA_DIR, 'bm25_index')
CHROMA_DIR = os.path.join(DATA_DIR, 'chroma')
CHROMA_COLLECTION_NAME = "hybridrank_elements"

BATCH_SIZE = 5000


def load_elements(path: str) -> List[Dict[str, Any]]:
    """Carga todos los elementos desde el archivo JSON de embeddings."""
    logger.info(f"Cargando elementos desde {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        elements = json.load(f)
    logger.info(f"Cargados {len(elements)} elementos.")

    valid = []
    skipped = 0
    for elem in elements:
        chunk_id = elem.get('metadata', {}).get('chunk_id')
        cleaned = elem.get('cleaned_content', '').strip()
        embedding = elem.get('embedding', [])
        if not chunk_id or not cleaned:
            skipped += 1
            continue
        if not embedding or len(embedding) == 0:
            skipped += 1
            continue
        valid.append(elem)

    if skipped > 0:
        logger.warning(f"Omitidos {skipped} elementos sin chunk_id, cleaned_content o embedding.")
    logger.info(f"{len(valid)} elementos validos para indexacion.")
    return valid


def build_bm25_index(elements: List[Dict[str, Any]], index_dir: str) -> None:
    """
    Construye un indice BM25 usando rank-bm25 y lo persiste en disco.

    :param elements: Lista de elementos con 'metadata.chunk_id' y 'cleaned_content'.
    :param index_dir: Directorio donde guardar el indice.
    """
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
    os.makedirs(index_dir, exist_ok=True)

    tokenizer = SpanishTokenizer()

    logger.info("Tokenizando corpus para BM25...")
    doc_ids = []
    tokenized_corpus = []

    for elem in elements:
        chunk_id = elem['metadata']['chunk_id']
        content = elem['cleaned_content']
        tokens = tokenizer.tokenize(content)

        if tokens:
            doc_ids.append(chunk_id)
            tokenized_corpus.append(tokens)

    logger.info(f"Corpus tokenizado: {len(tokenized_corpus)} documentos.")

    logger.info("Construyendo modelo BM25Okapi...")
    bm25 = BM25Okapi(tokenized_corpus)
    logger.info(f"Modelo BM25 construido: corpus_size={bm25.corpus_size}, avgdl={bm25.avgdl:.2f}")

    model_path = os.path.join(index_dir, 'bm25_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(bm25, f)
    logger.info(f"Modelo BM25 guardado en {model_path}")

    doc_ids_path = os.path.join(index_dir, 'doc_ids.json')
    with open(doc_ids_path, 'w', encoding='utf-8') as f:
        json.dump(doc_ids, f, ensure_ascii=False)
    logger.info(f"Mapeo de IDs guardado: {len(doc_ids)} documentos en {doc_ids_path}")

    metadata_path = os.path.join(index_dir, 'metadata.json')
    metadata = {
        "num_docs": len(doc_ids),
        "avg_doc_length": float(bm25.avgdl),
        "k1": bm25.k1,
        "b": bm25.b,
        "tokenizer": "es_core_news_md",
        "created_at": datetime.now().isoformat()
    }
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info("Indice BM25 construido exitosamente.")


def _sanitize_metadata(meta: dict) -> dict:
    """Asegura que todos los valores de metadata sean compatibles con ChromaDB (no None)."""
    clean = {}
    for key in ['source', 'type', 'filename', 'document_type', 'filetype']:
        clean[key] = meta.get(key) or ""
    clean['page_number'] = meta.get('page_number') if meta.get('page_number') is not None else -1
    return clean


def build_chroma_index(elements: List[Dict[str, Any]], chroma_dir: str, collection_name: str) -> None:
    """
    Carga embeddings y metadata en ChromaDB con persistencia en disco.
    Usa distancia coseno para busqueda por similaridad.
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

    total = len(elements)
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = elements[start:end]

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for elem in batch:
            chunk_id = elem['metadata']['chunk_id']
            ids.append(chunk_id)
            embeddings.append(elem['embedding'])
            documents.append(elem['cleaned_content'])
            metadatas.append(_sanitize_metadata(elem['metadata']))

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        logger.info(f"ChromaDB: insertados {end}/{total} elementos.")

    logger.info(f"Verificacion ChromaDB: {collection.count()} documentos en coleccion.")


def main():
    logger.info("=" * 60)
    logger.info("Inicio de indexacion HybridRank")
    logger.info("=" * 60)

    if not os.path.exists(EMBEDDINGS_PATH):
        logger.error(f"No se encontro el archivo de embeddings: {EMBEDDINGS_PATH}")
        sys.exit(1)

    elements = load_elements(EMBEDDINGS_PATH)

    logger.info("\n--- Paso 1/2: Construyendo indice BM25 (rank-bm25) ---")
    build_bm25_index(elements, BM25_INDEX_DIR)

    logger.info("\n--- Paso 2/2: Construyendo indice ChromaDB (Dense) ---")
    build_chroma_index(elements, CHROMA_DIR, CHROMA_COLLECTION_NAME)

    logger.info("\n" + "=" * 60)
    logger.info("Indexacion completada exitosamente.")
    logger.info(f"  BM25 index:  {BM25_INDEX_DIR}")
    logger.info(f"  ChromaDB:    {CHROMA_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
