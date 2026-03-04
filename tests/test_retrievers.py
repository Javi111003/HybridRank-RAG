"""
Tests unitarios para BM25Retriever y DenseRetriever.

Usan mocks para evitar dependencias de indices en disco,
modelos spaCy, ChromaDB y SentenceTransformer.
"""

import unittest
from unittest.mock import patch, MagicMock, mock_open, call
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bm25_mock(scores=None, k1=1.5, b=0.75):
    """Crea un mock de BM25Okapi con los atributos minimos necesarios."""
    mock = MagicMock()
    mock.get_scores.return_value = np.array(scores if scores is not None else [1.2, 0.5, 0.0])
    mock.k1 = k1
    mock.b = b
    mock.corpus_size = 3
    mock.avgdl = 20.0
    return mock


def _make_tokenizer_mock(return_tokens=None):
    """Crea un mock de SpanishTokenizer."""
    mock = MagicMock()
    mock.tokenize.return_value = return_tokens if return_tokens is not None else ["consulta", "test"]
    return mock


# ---------------------------------------------------------------------------
# Tests BM25Retriever
# ---------------------------------------------------------------------------

class TestBM25RetrieverInit(unittest.TestCase):
    """Tests de inicializacion del BM25Retriever."""

    @patch('src.retriever.bm25_retriever.SpanishTokenizer')
    @patch('src.retriever.bm25_retriever.json.load')
    @patch('src.retriever.bm25_retriever.pickle.load')
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.retriever.bm25_retriever.os.path.exists')
    def test_init_carga_modelo_y_doc_ids(
        self, mock_exists, mock_file, mock_pickle, mock_json, mock_tokenizer_cls
    ):
        """Verifica que __init__ cargue el modelo BM25 y los IDs desde disco."""
        mock_exists.return_value = True
        mock_bm25 = _make_bm25_mock()
        mock_pickle.return_value = mock_bm25
        mock_json.return_value = ['chunk_001', 'chunk_002', 'chunk_003']
        mock_tokenizer_cls.return_value = _make_tokenizer_mock()

        from src.retriever.bm25_retriever import BM25Retriever
        retriever = BM25Retriever(index_dir='/fake/index')

        self.assertIs(retriever._bm25, mock_bm25)
        self.assertEqual(retriever._doc_ids, ['chunk_001', 'chunk_002', 'chunk_003'])
        self.assertEqual(retriever.name, 'BM25Retriever')

    @patch('src.retriever.bm25_retriever.os.path.exists')
    def test_init_lanza_error_si_falta_indice(self, mock_exists):
        """Verifica que se lanza FileNotFoundError si el modelo no existe en disco."""
        mock_exists.return_value = False

        from src.retriever.bm25_retriever import BM25Retriever
        with self.assertRaises(FileNotFoundError) as ctx:
            BM25Retriever(index_dir='/no/existe')

        error_msg = str(ctx.exception).lower()
        self.assertTrue(
            'bm25' in error_msg or '/no/existe' in error_msg,
            f"Mensaje de error inesperado: {ctx.exception}"
        )

    @patch('src.retriever.bm25_retriever.SpanishTokenizer')
    @patch('src.retriever.bm25_retriever.json.load')
    @patch('src.retriever.bm25_retriever.pickle.load')
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.retriever.bm25_retriever.os.path.exists')
    def test_init_usa_ruta_por_defecto_si_no_se_indica_directorio(
        self, mock_exists, mock_file, mock_pickle, mock_json, mock_tokenizer_cls
    ):
        """Verifica que index_dir por defecto apunte a .data/bm25_index."""
        mock_exists.return_value = True
        mock_pickle.return_value = _make_bm25_mock()
        mock_json.return_value = []
        mock_tokenizer_cls.return_value = _make_tokenizer_mock()

        from src.retriever.bm25_retriever import BM25Retriever
        retriever = BM25Retriever()  # sin argumentos

        # La ruta chequeada debe contener bm25_index
        checked_path = mock_exists.call_args[0][0]
        self.assertIn('bm25_index', checked_path)


# ---------------------------------------------------------------------------

class TestBM25RetrieverRetrieve(unittest.TestCase):
    """Tests del metodo retrieve del BM25Retriever."""

    def _build_retriever(self, scores, doc_ids, query_tokens=None):
        """
        Construye un BM25Retriever completamente mockeado sin tocar disco.
        Retorna la instancia lista para usar en tests.
        """
        with patch('src.retriever.bm25_retriever.os.path.exists', return_value=True), \
             patch('builtins.open', mock_open()), \
             patch('src.retriever.bm25_retriever.pickle.load', return_value=_make_bm25_mock(scores)), \
             patch('src.retriever.bm25_retriever.json.load', return_value=doc_ids), \
             patch('src.retriever.bm25_retriever.SpanishTokenizer') as mock_tok_cls:

            mock_tok_cls.return_value = _make_tokenizer_mock(query_tokens or ["ley", "decreto"])

            from src.retriever.bm25_retriever import BM25Retriever
            retriever = BM25Retriever(index_dir='/fake/index')

        return retriever

    def test_retrieve_retorna_lista_de_tuplas(self):
        """El resultado debe ser una lista de (chunk_id, score)."""
        retriever = self._build_retriever(
            scores=[1.2, 0.5, 0.0],
            doc_ids=['doc_a', 'doc_b', 'doc_c']
        )
        results = retriever.retrieve("ley de educacion", top_k=3)

        self.assertIsInstance(results, list)
        for item in results:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
            self.assertIsInstance(item[0], str)
            self.assertIsInstance(item[1], float)

    def test_retrieve_ordena_por_score_descendente(self):
        """Los resultados deben estar ordenados de mayor a menor score."""
        retriever = self._build_retriever(
            scores=[0.5, 1.8, 0.3],
            doc_ids=['doc_a', 'doc_b', 'doc_c']
        )
        results = retriever.retrieve("consulta legal", top_k=3)

        scores = [r[1] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_retrieve_excluye_scores_cero(self):
        """Los documentos con score == 0.0 no deben aparecer en los resultados."""
        retriever = self._build_retriever(
            scores=[1.2, 0.0, 0.0],
            doc_ids=['doc_a', 'doc_b', 'doc_c']
        )
        results = retriever.retrieve("consulta", top_k=3)

        ids = [r[0] for r in results]
        self.assertNotIn('doc_b', ids)
        self.assertNotIn('doc_c', ids)
        self.assertIn('doc_a', ids)

    def test_retrieve_respeta_top_k(self):
        """No debe retornar mas de top_k resultados."""
        retriever = self._build_retriever(
            scores=[2.0, 1.5, 1.0],
            doc_ids=['doc_a', 'doc_b', 'doc_c']
        )
        results = retriever.retrieve("ley", top_k=2)

        self.assertLessEqual(len(results), 2)

    def test_retrieve_devuelve_vacio_si_query_tokenizada_vacia(self):
        """Si la query tokeniza a lista vacia, debe retornar [] sin llamar a BM25."""
        retriever = self._build_retriever(
            scores=[1.0, 0.8],
            doc_ids=['doc_a', 'doc_b'],
            query_tokens=[]      # tokenizador devolvera lista vacia
        )
        # Sobreescribir el tokenizador del retriever para simular query vacia
        retriever._tokenizer.tokenize.return_value = []

        results = retriever.retrieve("   ", top_k=5)

        self.assertEqual(results, [])
        retriever._bm25.get_scores.assert_not_called()

    def test_retrieve_top_k_mayor_que_corpus_no_falla(self):
        """Si top_k > numero de documentos, debe retornar todos los disponibles."""
        retriever = self._build_retriever(
            scores=[1.0, 0.5],
            doc_ids=['doc_a', 'doc_b']
        )
        results = retriever.retrieve("decreto", top_k=100)

        self.assertLessEqual(len(results), 2)


# ---------------------------------------------------------------------------

class TestBM25RetrieverParams(unittest.TestCase):
    """Tests de set_bm25_parameters."""

    def _build_retriever(self):
        with patch('src.retriever.bm25_retriever.os.path.exists', return_value=True), \
             patch('builtins.open', mock_open()), \
             patch('src.retriever.bm25_retriever.pickle.load', return_value=_make_bm25_mock()), \
             patch('src.retriever.bm25_retriever.json.load', return_value=['doc1']), \
             patch('src.retriever.bm25_retriever.SpanishTokenizer') as mock_cls:
            mock_cls.return_value = _make_tokenizer_mock()
            from src.retriever.bm25_retriever import BM25Retriever
            return BM25Retriever(index_dir='/fake/index')

    def test_set_bm25_parameters_actualiza_k1_y_b(self):
        """Verifica que k1 y b se actualicen en el modelo BM25."""
        retriever = self._build_retriever()
        retriever.set_bm25_parameters(k1=2.0, b=0.9)

        self.assertEqual(retriever._bm25.k1, 2.0)
        self.assertEqual(retriever._bm25.b, 0.9)

    def test_set_bm25_parameters_valores_extremos(self):
        """Verifica asignacion con valores limite."""
        retriever = self._build_retriever()
        retriever.set_bm25_parameters(k1=0.0, b=0.0)

        self.assertEqual(retriever._bm25.k1, 0.0)
        self.assertEqual(retriever._bm25.b, 0.0)


# ---------------------------------------------------------------------------
# Tests DenseRetriever
# ---------------------------------------------------------------------------

class TestDenseRetrieverInit(unittest.TestCase):
    """Tests de inicializacion del DenseRetriever."""

    @patch('src.retriever.dense_retriever.SentenceTransformer')
    @patch('src.retriever.dense_retriever.chromadb.PersistentClient')
    @patch('src.retriever.dense_retriever.os.path.exists')
    def test_init_conecta_coleccion_y_carga_modelo(
        self, mock_exists, mock_chroma_cls, mock_st_cls
    ):
        """Verifica que __init__ cree el cliente ChromaDB y cargue el modelo."""
        mock_exists.return_value = True
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_chroma_cls.return_value = mock_client
        mock_st_cls.return_value = MagicMock()

        from src.retriever.dense_retriever import DenseRetriever
        retriever = DenseRetriever(chroma_dir='/fake/chroma')

        mock_chroma_cls.assert_called_once_with(path='/fake/chroma')
        mock_client.get_collection.assert_called_once_with(name='hybridrank_elements')
        self.assertEqual(retriever.name, 'DenseRetriever')

    @patch('src.retriever.dense_retriever.os.path.exists')
    def test_init_lanza_error_si_falta_chroma_dir(self, mock_exists):
        """Verifica FileNotFoundError cuando el directorio ChromaDB no existe."""
        mock_exists.return_value = False

        from src.retriever.dense_retriever import DenseRetriever
        with self.assertRaises(FileNotFoundError) as ctx:
            DenseRetriever(chroma_dir='/no/chroma')

        self.assertIn('/no/chroma', str(ctx.exception))

    @patch('src.retriever.dense_retriever.SentenceTransformer')
    @patch('src.retriever.dense_retriever.chromadb.PersistentClient')
    @patch('src.retriever.dense_retriever.os.path.exists')
    def test_init_permite_coleccion_personalizada(
        self, mock_exists, mock_chroma_cls, mock_st_cls
    ):
        """Verifica que collection_name personalizado se pase a get_collection."""
        mock_exists.return_value = True
        mock_client = MagicMock()
        mock_chroma_cls.return_value = mock_client
        mock_client.get_collection.return_value = MagicMock()
        mock_st_cls.return_value = MagicMock()

        from src.retriever.dense_retriever import DenseRetriever
        DenseRetriever(chroma_dir='/fake/chroma', collection_name='mi_coleccion')

        mock_client.get_collection.assert_called_once_with(name='mi_coleccion')

    @patch('src.retriever.dense_retriever.SentenceTransformer')
    @patch('src.retriever.dense_retriever.chromadb.PersistentClient')
    @patch('src.retriever.dense_retriever.os.path.exists')
    def test_init_usa_ruta_por_defecto(self, mock_exists, mock_chroma_cls, mock_st_cls):
        """Verifica que chroma_dir por defecto apunte a .data/chroma."""
        mock_exists.return_value = True
        mock_chroma_cls.return_value = MagicMock()
        mock_chroma_cls.return_value.get_collection.return_value = MagicMock()
        mock_st_cls.return_value = MagicMock()

        from src.retriever.dense_retriever import DenseRetriever
        DenseRetriever()

        checked_path = mock_exists.call_args[0][0]
        self.assertIn('chroma', checked_path)


# ---------------------------------------------------------------------------

class TestDenseRetrieverRetrieve(unittest.TestCase):
    """Tests del metodo retrieve del DenseRetriever."""

    def _build_retriever(self, ids=None, distances=None):
        """Construye un DenseRetriever mockeado listo para retrieve."""
        if ids is None:
            ids = ['chunk_a', 'chunk_b', 'chunk_c']
        if distances is None:
            distances = [0.1, 0.3, 0.6]

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            'ids': [ids],
            'distances': [distances]
        }

        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])

        with patch('src.retriever.dense_retriever.os.path.exists', return_value=True), \
             patch('src.retriever.dense_retriever.chromadb.PersistentClient', return_value=mock_client), \
             patch('src.retriever.dense_retriever.SentenceTransformer', return_value=mock_model):

            from src.retriever.dense_retriever import DenseRetriever
            retriever = DenseRetriever(chroma_dir='/fake/chroma')

        return retriever

    def test_retrieve_retorna_lista_de_tuplas(self):
        """El resultado debe ser lista de (chunk_id, similarity)."""
        retriever = self._build_retriever()
        results = retriever.retrieve("resolucion ministerial", top_k=3)

        self.assertIsInstance(results, list)
        for item in results:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
            self.assertIsInstance(item[0], str)
            self.assertIsInstance(item[1], float)

    def test_retrieve_convierte_distancia_a_similaridad(self):
        """similarity debe ser 1.0 - distance (distancia coseno de ChromaDB)."""
        retriever = self._build_retriever(
            ids=['doc_x', 'doc_y'],
            distances=[0.2, 0.8]
        )
        results = retriever.retrieve("norma juridica", top_k=2)

        result_dict = dict(results)
        self.assertAlmostEqual(result_dict['doc_x'], 1.0 - 0.2)
        self.assertAlmostEqual(result_dict['doc_y'], 1.0 - 0.8)

    def test_retrieve_llama_encode_con_la_query(self):
        """Verifica que la query se pase al modelo de embeddings."""
        retriever = self._build_retriever()
        retriever.retrieve("decreto ley", top_k=2)

        retriever._model.encode.assert_called_once_with("decreto ley")

    def test_retrieve_pasa_top_k_a_chromadb(self):
        """Verifica que n_results de ChromaDB sea igual a top_k."""
        retriever = self._build_retriever()
        retriever.retrieve("consulta", top_k=7)

        retriever._collection.query.assert_called_once()
        call_kwargs = retriever._collection.query.call_args[1]
        self.assertEqual(call_kwargs['n_results'], 7)

    def test_retrieve_preserva_orden_de_chromadb(self):
        """El orden de los resultados debe coincidir con el retornado por ChromaDB."""
        ids = ['doc_1', 'doc_2', 'doc_3']
        distances = [0.05, 0.25, 0.55]
        retriever = self._build_retriever(ids=ids, distances=distances)
        results = retriever.retrieve("ordenanza", top_k=3)

        returned_ids = [r[0] for r in results]
        self.assertEqual(returned_ids, ids)

    def test_retrieve_embedding_se_pasa_como_lista(self):
        """El embedding de la query debe enviarse como lista a ChromaDB."""
        retriever = self._build_retriever()
        retriever.retrieve("articulo", top_k=2)

        call_kwargs = retriever._collection.query.call_args[1]
        self.assertIsInstance(call_kwargs['query_embeddings'][0], list)


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main(verbosity=2)
