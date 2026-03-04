import os
import json
import pickle
import logging
from typing import List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from .retriever import Retriever
from .tokenizer import SpanishTokenizer

logger = logging.getLogger(__name__)


class BM25Retriever(Retriever):
    """
    Recuperacion dispersa usando BM25 (rank-bm25) sobre un indice persistido en disco.
    El indice debe ser construido previamente por index_builder.py.
    """

    def __init__(self, index_dir: str = None):
        """
        :param index_dir: Ruta al directorio del indice BM25.
                          Por defecto: .data/bm25_index relativo a la raiz del proyecto.
        """
        if index_dir is None:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            index_dir = os.path.join(project_root, '.data', 'bm25_index')

        model_path = os.path.join(index_dir, 'bm25_model.pkl')
        doc_ids_path = os.path.join(index_dir, 'doc_ids.json')

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No se encontro el indice BM25 en {index_dir}. "
                "Ejecuta primero: python -m src.indexing.index_builder"
            )

        with open(model_path, 'rb') as f:
            self._bm25: BM25Okapi = pickle.load(f)

        with open(doc_ids_path, 'r', encoding='utf-8') as f:
            self._doc_ids: List[str] = json.load(f)

        self._tokenizer = SpanishTokenizer()

        logger.info(f"BM25Retriever cargado: {len(self._doc_ids)} documentos desde {index_dir}")

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Busca en el indice BM25.

        :param query: Texto de la consulta.
        :param top_k: Numero de resultados a retornar.
        :return: Lista de (chunk_id, score_bm25) ordenada por score descendente.
        """
        tokenized_query = self._tokenizer.tokenize(query)

        if not tokenized_query:
            logger.warning(f"Query tokenizada vacia para: '{query[:50]}...'")
            return []

        scores = self._bm25.get_scores(tokenized_query)

        top_k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0.0:
                results.append((self._doc_ids[idx], score))

        return results

    @property
    def name(self) -> str:
        return "BM25Retriever"

    def set_bm25_parameters(self, k1: float, b: float) -> None:
        """
        Ajusta los parametros de BM25 para experimentacion.
        k1 y b se aplican en el calculo de score en tiempo de query.

        :param k1: Parametro de saturacion de frecuencia de termino (default: 1.5).
        :param b: Parametro de normalizacion por longitud de documento (default: 0.75).
        """
        self._bm25.k1 = k1
        self._bm25.b = b
        logger.info(f"Parametros BM25 actualizados: k1={k1}, b={b}")
