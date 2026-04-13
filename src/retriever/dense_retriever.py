import os
from typing import List, Tuple

import chromadb
from sentence_transformers import SentenceTransformer

from src.indexing.index_builder import (
    CHROMA_DIR as DEFAULT_CHROMA_DIR,
    CHROMA_COLLECTION_NAME as DEFAULT_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME as DEFAULT_EMBEDDING_MODEL,
)
from .retriever import Retriever


class DenseRetriever(Retriever):
    """
    Recuperacion densa usando ChromaDB con embeddings pre-computados.
    Usa similaridad coseno para busqueda de vecinos mas cercanos.
    La coleccion ChromaDB debe ser construida previamente por index_builder.py.

    Usa el modelo E5 multilingual con prefijo 'query: ' para codificar queries,
    complementando el prefijo 'passage: ' usado en la indexacion de documentos.
    """

    QUERY_PROMPT = "query: "

    def __init__(
        self,
        chroma_dir: str = None,
        collection_name: str = None,
        embedding_model_name: str = None
    ):
        """
        :param chroma_dir: Ruta al almacenamiento persistente de ChromaDB.
                           Por defecto usa la ruta definida en index_builder.
        :param collection_name: Nombre de la coleccion en ChromaDB.
        :param embedding_model_name: Modelo SentenceTransformer para codificar queries.
        """
        chroma_dir = chroma_dir or DEFAULT_CHROMA_DIR
        collection_name = collection_name or DEFAULT_COLLECTION_NAME

        if not os.path.exists(chroma_dir):
            raise FileNotFoundError(
                f"No se encontro el almacenamiento ChromaDB en {chroma_dir}. "
                "Ejecuta primero: python -m src.indexing.index_builder"
            )

        self._client = chromadb.PersistentClient(path=chroma_dir)
        self._collection = self._client.get_collection(name=collection_name)
        self._model = SentenceTransformer(
            embedding_model_name or DEFAULT_EMBEDDING_MODEL
        )

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Busca en ChromaDB usando similaridad coseno.

        ChromaDB con espacio coseno retorna distancias en [0, 2].
        Convertimos a similaridad: similarity = 1 - distance, rango [-1, 1].

        :param query: Texto de la consulta.
        :param top_k: Numero de resultados a retornar.
        :return: Lista de (chunk_id, similaridad_coseno) ordenada por similaridad descendente.
        """
        query_embedding = self._model.encode(query, prompt=self.QUERY_PROMPT).tolist()

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        ids = results['ids'][0]
        distances = results['distances'][0]

        return [(doc_id, 1.0 - distance) for doc_id, distance in zip(ids, distances)]

    @property
    def name(self) -> str:
        return "DenseRetriever"
