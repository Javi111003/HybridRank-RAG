"""Tests for index_builder metadata and norma-compatible indexing."""

import json
import os
import pickle
import tempfile
import unittest
from unittest.mock import patch

from src.indexing import index_builder


class _WhitespaceTokenizer:
    def tokenize_batch(self, texts, batch_size=1000, n_process=1):
        return [text.lower().split() for text in texts]


class TestIndexBuilderMetadata(unittest.TestCase):

    def test_sanitize_metadata_preserves_norma_fragment_fields(self):
        metadata = {
            "chunk_id": "frag-1",
            "source": "normas.db",
            "type": "NormaIndexFragment",
            "document_type": "norma",
            "corpus_type": "normas",
            "norma_id": "resolucion_1_2026_ministerio_test",
            "fragment_id": "frag-1",
            "fragment_index": 0,
            "fragment_total": 2,
            "fragment_label": "ARTICULO 1",
            "tipo": "Resolucion",
            "numero": "1",
            "year": 2026,
            "organismo_emisor": "Ministerio Test",
            "goc_code": "GOC-2026-001-O1",
            "gaceta_normas": ["Resolucion 1 de 2026 de Ministerio Test"],
            "ignored_complex": {"x": 1},
        }

        sanitized = index_builder._sanitize_metadata(metadata)

        self.assertEqual(sanitized["corpus_type"], "normas")
        self.assertEqual(sanitized["norma_id"], metadata["norma_id"])
        self.assertEqual(sanitized["fragment_label"], "ARTICULO 1")
        self.assertEqual(sanitized["gaceta_normas"], json.dumps(metadata["gaceta_normas"], ensure_ascii=False))
        self.assertNotIn("ignored_complex", sanitized)


class TestIndexBuilderSmoke(unittest.TestCase):

    def _write_fake_embeddings(self, path):
        docs = [
            {
                "content": "Resolucion 1 de 2026. ARTICULO 1. texto uno",
                "cleaned_content": "resolucion articulo texto uno",
                "embedding": [0.1, 0.2, 0.3],
                "metadata": {
                    "chunk_id": "frag-1",
                    "corpus_type": "normas",
                    "norma_id": "norma_1",
                    "fragment_id": "frag-1",
                },
            },
            {
                "content": "Resolucion 2 de 2026. ARTICULO 2. texto dos",
                "cleaned_content": "resolucion articulo texto dos",
                "embedding": [0.2, 0.1, 0.4],
                "metadata": {
                    "chunk_id": "frag-2",
                    "corpus_type": "normas",
                    "norma_id": "norma_2",
                    "fragment_id": "frag-2",
                },
            },
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(docs, f)

    def test_builds_bm25_and_chroma_from_fake_norma_embeddings(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            input_path = os.path.join(tmpdir, "fake_embeddings.json")
            bm25_dir = os.path.join(tmpdir, "bm25")
            chroma_dir = os.path.join(tmpdir, "chroma")
            self._write_fake_embeddings(input_path)

            with patch("src.indexing.index_builder._create_spanish_tokenizer", return_value=_WhitespaceTokenizer()):
                index_builder.build_bm25_index(input_path, bm25_dir, batch_size=1)

            self.assertTrue(os.path.exists(os.path.join(bm25_dir, "bm25_model.pkl")))
            with open(os.path.join(bm25_dir, "doc_ids.json"), "r", encoding="utf-8") as f:
                self.assertEqual(json.load(f), ["frag-1", "frag-2"])
            with open(os.path.join(bm25_dir, "bm25_model.pkl"), "rb") as f:
                bm25 = pickle.load(f)
            self.assertEqual(bm25.corpus_size, 2)

            index_builder.build_chroma_index(input_path, chroma_dir, "test_normas", batch_size=1)
            client = index_builder.chromadb.PersistentClient(path=chroma_dir)
            collection = client.get_collection("test_normas")
            result = collection.get(ids=["frag-1"], include=["documents", "metadatas"])

            self.assertEqual(collection.count(), 2)
            self.assertIn("ARTICULO 1", result["documents"][0])
            self.assertEqual(result["metadatas"][0]["norma_id"], "norma_1")


if __name__ == '__main__':
    unittest.main(verbosity=2)
