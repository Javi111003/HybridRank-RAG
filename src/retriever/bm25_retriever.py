import json
import logging
import os
import pickle
from os import PathLike
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .retriever import Retriever
from .tokenizer import SpanishTokenizer
from .types import RetrievalResults

logger = logging.getLogger(__name__)

DEFAULT_INDEX_DIR = Path(__file__).resolve().parents[2] / ".data" / "bm25_index"


class BM25Retriever(Retriever):
    """Sparse retriever backed by a persisted rank-bm25 index."""

    def __init__(self, index_dir: str | PathLike[str] | None = None):
        self._index_dir = Path(index_dir) if index_dir is not None else DEFAULT_INDEX_DIR

        model_path = self._index_dir / "bm25_model.pkl"
        doc_ids_path = self._index_dir / "doc_ids.json"

        if not os.path.exists(str(model_path)):
            raise FileNotFoundError(
                f"No se encontro el indice BM25 en {self._index_dir}. "
                "Ejecuta primero: python -m src.indexing.index_builder"
            )

        with open(model_path, "rb") as f:
            self._bm25: BM25Okapi = pickle.load(f)

        with open(doc_ids_path, "r", encoding="utf-8") as f:
            self._doc_ids: list[str] = json.load(f)

        self._tokenizer = SpanishTokenizer()

        logger.info(
            "BM25Retriever cargado: %d documentos desde %s",
            len(self._doc_ids),
            self._index_dir,
        )

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResults:
        if top_k <= 0:
            return []

        tokenized_query = self._tokenizer.tokenize(query)

        if not tokenized_query:
            logger.warning("Query tokenizada vacia: %r", query[:50])
            return []

        scores = self._bm25.get_scores(tokenized_query)
        top_k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: RetrievalResults = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0.0:
                results.append((self._doc_ids[idx], score))

        return results

    @property
    def name(self) -> str:
        return "BM25Retriever"

    def set_bm25_parameters(self, k1: float, b: float) -> None:
        self._bm25.k1 = k1
        self._bm25.b = b
        logger.info("Parametros BM25 actualizados: k1=%s, b=%s", k1, b)
