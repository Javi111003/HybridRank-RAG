import pytest
from unittest.mock import MagicMock, patch

from src.rag.store.norma_store import NormaStore
from src.rag.store.models import RetrievedFragment


def _mock_chroma_collection(docs_data):
    """Crea un mock de coleccion ChromaDB con datos predefinidos."""
    collection = MagicMock()
    collection.count.return_value = len(docs_data)

    def mock_get(ids, include=None):
        result_ids = []
        result_docs = []
        result_metas = []
        for fid in ids:
            if fid in docs_data:
                result_ids.append(fid)
                result_docs.append(docs_data[fid]["document"])
                result_metas.append(docs_data[fid]["metadata"])
        return {"ids": result_ids, "documents": result_docs, "metadatas": result_metas}

    collection.get = MagicMock(side_effect=mock_get)
    return collection


SAMPLE_DOCS = {
    "frag_001": {
        "document": "Texto del articulo 1 sobre licencias.",
        "metadata": {
            "norma_id": "resolucion_41_2025_mtss",
            "tipo": "Resolucion",
            "numero": "41",
            "year": 2025,
            "organismo_emisor": "MTSS",
            "goc_code": "GOC-2026-100-O10",
            "gaceta_numero": "10",
            "gaceta_fecha": "15 Enero, 2026",
            "page_start": 5,
            "page_end": 12,
        },
    },
    "frag_002": {
        "document": "Texto del articulo 2 sobre prestaciones.",
        "metadata": {
            "norma_id": "resolucion_41_2025_mtss",
            "tipo": "Resolucion",
            "numero": "41",
            "year": 2025,
            "organismo_emisor": "MTSS",
            "goc_code": "GOC-2026-100-O10",
            "gaceta_numero": "10",
            "gaceta_fecha": "15 Enero, 2026",
            "page_start": 5,
            "page_end": 12,
        },
    },
}


class TestNormaStore:
    @patch("src.rag.store.norma_store.chromadb")
    def test_get_fragments_returns_ordered(self, mock_chromadb):
        mock_client = MagicMock()
        mock_collection = _mock_chroma_collection(SAMPLE_DOCS)
        mock_client.get_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        store = NormaStore(chroma_dir="/fake/path", collection_name="test")

        results = [("frag_001", 0.9), ("frag_002", 0.7)]
        fragments = store.get_fragments(results)

        assert len(fragments) == 2
        assert fragments[0].fragment_id == "frag_001"
        assert fragments[0].score == 0.9
        assert fragments[0].rank == 1
        assert fragments[1].fragment_id == "frag_002"
        assert fragments[1].score == 0.7
        assert fragments[1].rank == 2

    @patch("src.rag.store.norma_store.chromadb")
    def test_get_fragments_handles_missing_ids(self, mock_chromadb):
        mock_client = MagicMock()
        mock_collection = _mock_chroma_collection(SAMPLE_DOCS)
        mock_client.get_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        store = NormaStore(chroma_dir="/fake/path", collection_name="test")

        results = [("frag_001", 0.9), ("nonexistent", 0.5)]
        fragments = store.get_fragments(results)

        assert len(fragments) == 1
        assert fragments[0].fragment_id == "frag_001"

    @patch("src.rag.store.norma_store.chromadb")
    def test_get_fragments_empty_input(self, mock_chromadb):
        mock_client = MagicMock()
        mock_client.get_collection.return_value = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client

        store = NormaStore(chroma_dir="/fake/path", collection_name="test")
        assert store.get_fragments([]) == []

    @patch("src.rag.store.norma_store.chromadb")
    def test_get_fragments_metadata_mapped(self, mock_chromadb):
        mock_client = MagicMock()
        mock_collection = _mock_chroma_collection(SAMPLE_DOCS)
        mock_client.get_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        store = NormaStore(chroma_dir="/fake/path", collection_name="test")

        fragments = store.get_fragments([("frag_001", 0.9)])
        frag = fragments[0]

        assert frag.tipo == "Resolucion"
        assert frag.numero == "41"
        assert frag.year == 2025
        assert frag.organismo_emisor == "MTSS"
        assert frag.content == "Texto del articulo 1 sobre licencias."

    @patch("src.rag.store.norma_store.chromadb")
    def test_get_fragment_single(self, mock_chromadb):
        mock_client = MagicMock()
        mock_collection = _mock_chroma_collection(SAMPLE_DOCS)
        mock_client.get_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        store = NormaStore(chroma_dir="/fake/path", collection_name="test")
        frag = store.get_fragment("frag_001")

        assert frag is not None
        assert frag.fragment_id == "frag_001"

    @patch("src.rag.store.norma_store.chromadb")
    def test_get_fragment_not_found(self, mock_chromadb):
        mock_client = MagicMock()
        mock_collection = _mock_chroma_collection(SAMPLE_DOCS)
        mock_client.get_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        store = NormaStore(chroma_dir="/fake/path", collection_name="test")
        assert store.get_fragment("nonexistent") is None
