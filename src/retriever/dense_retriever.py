import os
from typing import List, Tuple, Optional

import chromadb
from sentence_transformers import SentenceTransformer

from .retriever import Retriever


class DenseRetriever(Retriever):
    """
    Recuperacion densa usando ChromaDB con embeddings pre-computados.
    Usa similaridad coseno para busqueda de vecinos mas cercanos.
    La coleccion ChromaDB debe ser construida previamente por index_builder.py.
    """

    EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(
        self,
        chroma_dir: str = None,
        collection_name: str = "hybridrank_elements",
        embedding_model_name: str = None
    ):
        """
        :param chroma_dir: Ruta al almacenamiento persistente de ChromaDB.
                           Por defecto: .data/chroma relativo a la raiz del proyecto.
        :param collection_name: Nombre de la coleccion en ChromaDB.
        :param embedding_model_name: Modelo SentenceTransformer para codificar queries.
        """
        if chroma_dir is None:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            chroma_dir = os.path.join(project_root, '.data', 'chroma')

        if not os.path.exists(chroma_dir):
            raise FileNotFoundError(
                f"No se encontro el almacenamiento ChromaDB en {chroma_dir}. "
                "Ejecuta primero: python -m src.indexing.index_builder"
            )

        self._client = chromadb.PersistentClient(path=chroma_dir)
        self._collection = self._client.get_collection(name=collection_name)
        self._model = SentenceTransformer(
            embedding_model_name or self.EMBEDDING_MODEL_NAME
        )

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Busca en ChromaDB usando similaridad coseno.

        La query se embede en tiempo de ejecucion usando el mismo modelo
        SentenceTransformer que se uso para generar los embeddings almacenados.

        :param query: Texto de la consulta.
        :param top_k: Numero de resultados a retornar.
        :return: Lista de (chunk_id, similaridad_coseno) ordenada por similaridad descendente.
                 Scores en rango [-1, 1] donde 1 = identico.
        """
        query_embedding = self._model.encode(query).tolist()

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        ids = results['ids'][0]
        distances = results['distances'][0]

        output = []
        for doc_id, distance in zip(ids, distances):
            similarity = 1.0 - distance
            output.append((doc_id, similarity))

        return output

    @property
    def name(self) -> str:
        return "DenseRetriever"
