import asyncio
import unittest
from unittest.mock import patch

from src.data_preparation.document_loader import GacetaJsonDocumentLoader


class TestGacetaJsonDocumentLoader(unittest.TestCase):

    def _build_loader(self):
        return GacetaJsonDocumentLoader(
            base_data_dir="E:/fake_base",
            gaceta_json_path="E:/fake/gaceta_oficial.json",
            max_workers=1,
        )

    @patch("src.data_preparation.document_loader.spacy.load")
    def test_resolve_downloaded_pdf(self, _mock_spacy_load):
        loader = self._build_loader()
        entry = {
            "files": [
                {"path": "full/a.pdf", "status": "pending"},
                {"path": "full/b.pdf", "status": "downloaded", "checksum": "abc"},
            ]
        }

        file_info = loader._resolve_downloaded_pdf(entry)

        self.assertIsNotNone(file_info)
        self.assertEqual(file_info["path"], "full/b.pdf")

    @patch("src.data_preparation.document_loader.spacy.load")
    def test_enrich_with_gaceta_metadata(self, _mock_spacy_load):
        loader = self._build_loader()
        elements = [{"content": "texto", "metadata": {"chunk_id": "123"}}]
        entry = {
            "tipo_edicion": "Ordinaria",
            "fecha": "01 Enero, 2026",
            "numero": "1",
            "normas": ["Ley 1"],
            "pdf_url": "https://x/y.pdf",
        }
        file_info = {"checksum": "abc", "path": "full/y.pdf"}

        enriched = loader._enrich_with_gaceta_metadata(elements, entry, file_info, "E:/fake_base/full/y.pdf")

        self.assertEqual(enriched[0]["metadata"]["gaceta_tipo_edicion"], "Ordinaria")
        self.assertEqual(enriched[0]["metadata"]["gaceta_checksum"], "abc")
        self.assertEqual(enriched[0]["metadata"]["source"], "E:/fake_base/full/y.pdf")

    @patch("src.data_preparation.document_loader.spacy.load")
    @patch.object(GacetaJsonDocumentLoader, "_iter_gaceta_entries")
    @patch("src.data_preparation.document_loader.os.path.exists")
    @patch.object(GacetaJsonDocumentLoader, "load_document_async")
    def test_load_from_gaceta_index_filters_and_dedupes(
        self,
        mock_load_document_async,
        mock_exists,
        mock_iter_entries,
        _mock_spacy_load,
    ):
        loader = self._build_loader()

        mock_iter_entries.return_value = [
            {
                "tipo_edicion": "Ordinaria",
                "fecha": "01 Enero, 2026",
                "numero": "1",
                "normas": [],
                "pdf_url": "https://example.com/a.pdf",
                "files": [{"path": "full/a.pdf", "status": "downloaded", "checksum": "same"}],
            },
            {
                "tipo_edicion": "Ordinaria",
                "fecha": "01 Enero, 2026",
                "numero": "1",
                "normas": [],
                "pdf_url": "https://example.com/a2.pdf",
                "files": [{"path": "full/a2.pdf", "status": "downloaded", "checksum": "same"}],
            },
            {
                "tipo_edicion": "Extraordinaria",
                "fecha": "02 Enero, 2026",
                "numero": "2",
                "normas": [],
                "pdf_url": "https://example.com/b.pdf",
                "files": [{"path": "full/b.pdf", "status": "pending", "checksum": "bbb"}],
            },
        ]

        mock_exists.return_value = True
        mock_load_document_async.return_value = [{"content": "x", "metadata": {"chunk_id": "c1"}}]

        elements = asyncio.run(loader.load_from_gaceta_index_async(dedupe_by_checksum=True))

        self.assertEqual(len(elements), 1)
        self.assertEqual(elements[0]["metadata"]["gaceta_numero"], "1")


if __name__ == "__main__":
    unittest.main()
