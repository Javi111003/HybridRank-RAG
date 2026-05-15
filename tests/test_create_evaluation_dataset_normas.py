"""Tests for norma corpus support in create_evaluation_dataset.py."""

import os
import unittest
from unittest.mock import MagicMock, patch

from scripts import create_evaluation_dataset as dataset


class TestEvaluationDatasetNormas(unittest.TestCase):

    def test_resolve_corpus_config_uses_norma_defaults(self):
        config = dataset.resolve_corpus_config("normas")

        self.assertTrue(config["output"].endswith(os.path.join(".data", "evaluation", "norma_qrels.json")))
        self.assertTrue(config["bm25_index_dir"].endswith(os.path.join(".data", "bm25_norma_index")))
        self.assertTrue(config["chroma_dir"].endswith(os.path.join(".data", "chroma_normas")))
        self.assertEqual(config["collection_name"], "hybridrank_normas")

    @patch("scripts.create_evaluation_dataset.DenseRetriever")
    @patch("scripts.create_evaluation_dataset.BM25Retriever")
    def test_create_retrievers_uses_explicit_norma_paths(self, mock_bm25_cls, mock_dense_cls):
        config = {
            "bm25_index_dir": "/tmp/bm25_normas",
            "chroma_dir": "/tmp/chroma_normas",
            "collection_name": "hybridrank_normas",
        }

        dataset.create_retrievers(config)

        mock_bm25_cls.assert_called_once_with(index_dir="/tmp/bm25_normas")
        mock_dense_cls.assert_called_once_with(
            chroma_dir="/tmp/chroma_normas",
            collection_name="hybridrank_normas",
        )

    def test_build_result_record_adds_norma_trace_without_breaking_qrels(self):
        query = {"query_id": "q1", "query_type": "literal", "query": "licencia"}
        doc_ids = ["frag-1", "frag-2", "frag-3"]
        relevant_docs = ["frag-2", "frag-3"]
        metadata = {
            "frag-1": {"norma_id": "norma_a", "fragment_index": 0},
            "frag-2": {"norma_id": "norma_b", "fragment_index": 0},
            "frag-3": {"norma_id": "norma_b", "fragment_index": 1},
        }

        record = dataset.build_result_record(
            query,
            doc_ids,
            relevant_docs,
            corpus="normas",
            doc_metadata=metadata,
        )

        self.assertEqual(record["pool_docs"], doc_ids)
        self.assertEqual(record["relevant_docs"], relevant_docs)
        self.assertEqual(record["pool_normas"], ["norma_a", "norma_b"])
        self.assertEqual(record["relevant_normas"], ["norma_b"])
        self.assertEqual(record["doc_metadata"]["frag-3"]["fragment_index"], 1)

    def test_fetch_doc_metadata_reads_chroma_metadata(self):
        collection = MagicMock()
        collection.get.return_value = {
            "ids": ["frag-1"],
            "metadatas": [{"norma_id": "norma_1"}],
        }

        result = dataset.fetch_doc_metadata(["frag-1"], collection)

        collection.get.assert_called_once_with(ids=["frag-1"], include=["metadatas"])
        self.assertEqual(result["frag-1"]["norma_id"], "norma_1")


if __name__ == '__main__':
    unittest.main(verbosity=2)
