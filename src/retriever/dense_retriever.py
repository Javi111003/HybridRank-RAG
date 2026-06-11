import os
from os import PathLike

import chromadb
from sentence_transformers import SentenceTransformer

from src.indexing.index_builder import (
    CHROMA_DIR as DEFAULT_CHROMA_DIR,
    CHROMA_COLLECTION_NAME as DEFAULT_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME as DEFAULT_EMBEDDING_MODEL,
)
from .retriever import Retriever
from .types import RetrievalResults


class DenseRetriever(Retriever):
    """Dense retriever backed by a persisted ChromaDB collection."""

    QUERY_PROMPT = "query: "

    def __init__(
        self,
        chroma_dir: str | PathLike[str] | None = None,
        collection_name: str | None = None,
        embedding_model_name: str | None = None,
    ):
        self._chroma_dir = os.fspath(chroma_dir or DEFAULT_CHROMA_DIR)
        self._collection_name = collection_name or DEFAULT_COLLECTION_NAME

        if not os.path.exists(self._chroma_dir):
            raise FileNotFoundError(
                f"No se encontro el almacenamiento ChromaDB en {self._chroma_dir}. "
                "Ejecuta primero: python -m src.indexing.index_builder"
            )

        self._client = chromadb.PersistentClient(path=self._chroma_dir)
        self._collection = self._client.get_collection(name=self._collection_name)
        self._model = SentenceTransformer(
            embedding_model_name or DEFAULT_EMBEDDING_MODEL
        )

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResults:
        if top_k <= 0:
            return []

        query_vector = self._model.encode(query, prompt=self.QUERY_PROMPT)
        query_embedding = (
            query_vector.tolist()
            if hasattr(query_vector, "tolist")
            else list(query_vector)
        )

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        ids = results["ids"][0]
        distances = results["distances"][0]

        return [(doc_id, 1.0 - distance) for doc_id, distance in zip(ids, distances)]

    @property
    def name(self) -> str:
        return "DenseRetriever"
