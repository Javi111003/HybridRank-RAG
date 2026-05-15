"""Tests for exporting structured normas into retrieval fragments."""

import os
import sqlite3
import tempfile
import unittest

from src.data_preparation.norma_index_document_builder import (
    build_fragments_for_norma,
    build_norma_fragments,
    split_norma_text,
)


def _make_row(**overrides):
    row = {
        "norma_row_id": 1,
        "norma_id": "resolucion_1_2026_ministerio_test",
        "tipo": "Resolucion",
        "numero": "1",
        "year": 2026,
        "organismo_emisor": "Ministerio Test",
        "goc_code": "GOC-2026-001-O1",
        "raw_text": "GOC-2026-001-O1 RESOLUCION 1/2026 ARTICULO 1. Texto breve.",
        "page_start": 1,
        "page_end": 2,
        "ordinal_position": 0,
        "match_confidence": "high",
        "raw_metadata_string": "Resolucion 1 de 2026 de Ministerio Test",
        "gaceta_id": 10,
        "gaceta_numero": "1",
        "gaceta_fecha": "01 Enero, 2026",
        "gaceta_tipo_edicion": "Ordinaria",
        "gaceta_pdf_url": "https://example.com/gaceta.pdf",
        "gaceta_checksum": "abcdef1234567890",
    }
    row.update(overrides)
    return row


def _create_norma_db(path):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE gacetas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            numero TEXT NOT NULL,
            fecha TEXT NOT NULL,
            tipo_edicion TEXT NOT NULL,
            pdf_url TEXT,
            checksum TEXT,
            sumario_text TEXT,
            norma_count INTEGER DEFAULT 0
        );

        CREATE TABLE normas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            norma_id TEXT NOT NULL,
            tipo TEXT NOT NULL,
            numero TEXT NOT NULL,
            year INTEGER NOT NULL,
            organismo_emisor TEXT NOT NULL,
            goc_code TEXT NOT NULL,
            raw_text TEXT NOT NULL,
            page_start INTEGER,
            page_end INTEGER,
            ordinal_position INTEGER DEFAULT 0,
            match_confidence TEXT DEFAULT 'high',
            gaceta_id INTEGER NOT NULL REFERENCES gacetas(id),
            raw_metadata_string TEXT
        );

        CREATE TABLE norma_duplicates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            norma_id TEXT NOT NULL,
            kept_gaceta_id INTEGER REFERENCES gacetas(id),
            superseded_gaceta_id INTEGER REFERENCES gacetas(id),
            notes TEXT
        );
        """
    )
    return conn


class TestNormaIndexDocumentBuilder(unittest.TestCase):

    def test_build_fragments_uses_stable_id_and_metadata(self):
        row = _make_row()

        first = build_fragments_for_norma(row)
        second = build_fragments_for_norma(row)

        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].fragment_id, second[0].fragment_id)
        self.assertEqual(
            first[0].fragment_id,
            "resolucion_1_2026_ministerio_test__abcdef123456__GOC-2026-001-O1__f000",
        )
        self.assertTrue(first[0].content.startswith("Resolucion 1 de 2026 - Ministerio Test."))
        self.assertEqual(first[0].metadata["chunk_id"], first[0].fragment_id)
        self.assertEqual(first[0].metadata["corpus_type"], "normas")
        self.assertEqual(first[0].metadata["fragment_total"], 1)
        self.assertEqual(first[0].metadata["norma_id"], row["norma_id"])

    def test_excludes_superseded_normas_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "normas.db")
            conn = _create_norma_db(db_path)
            try:
                old_gaceta = conn.execute(
                    "INSERT INTO gacetas (numero, fecha, tipo_edicion, checksum, norma_count) "
                    "VALUES ('1', '01 Enero, 2025', 'Ordinaria', 'oldchecksum123', 1)"
                ).lastrowid
                new_gaceta = conn.execute(
                    "INSERT INTO gacetas (numero, fecha, tipo_edicion, checksum, norma_count) "
                    "VALUES ('2', '01 Enero, 2026', 'Ordinaria', 'newchecksum456', 1)"
                ).lastrowid
                for gaceta_id, text in [
                    (old_gaceta, "RESOLUCION 1/2026 ARTICULO 1. version vieja"),
                    (new_gaceta, "RESOLUCION 1/2026 ARTICULO 1. version nueva"),
                ]:
                    conn.execute(
                        "INSERT INTO normas "
                        "(norma_id, tipo, numero, year, organismo_emisor, goc_code, raw_text, "
                        "page_start, page_end, ordinal_position, match_confidence, gaceta_id, raw_metadata_string) "
                        "VALUES (?, 'Resolucion', '1', 2026, 'Ministerio Test', 'GOC-2026-001-O1', "
                        "?, 1, 1, 0, 'high', ?, 'Resolucion 1 de 2026 de Ministerio Test')",
                        ("resolucion_1_2026_ministerio_test", text, gaceta_id),
                    )
                conn.execute(
                    "INSERT INTO norma_duplicates (norma_id, kept_gaceta_id, superseded_gaceta_id, notes) "
                    "VALUES ('resolucion_1_2026_ministerio_test', ?, ?, 'test')",
                    (new_gaceta, old_gaceta),
                )
                conn.commit()
            finally:
                conn.close()

            current_only = build_norma_fragments(db_path)
            all_versions = build_norma_fragments(db_path, include_superseded=True)

        self.assertEqual(len(current_only), 1)
        self.assertIn("newchecksum4", current_only[0].fragment_id)
        self.assertEqual(len(all_versions), 2)

    def test_large_article_falls_back_to_overlapping_windows(self):
        long_text = "ARTICULO 1. " + " ".join(f"palabra{i}" for i in range(900))

        fragments = split_norma_text(long_text)

        self.assertGreater(len(fragments), 1)
        self.assertTrue(all("ARTICULO 1" in label for label, _ in fragments))
        self.assertIn("palabra0", fragments[0][1])
        self.assertIn("palabra320", fragments[1][1])

    def test_large_norma_splits_by_legal_structure(self):
        article_1 = "ARTICULO 1. " + " ".join(f"a{i}" for i in range(240))
        article_2 = "ARTICULO 2. " + " ".join(f"b{i}" for i in range(240))

        fragments = split_norma_text(f"{article_1}. {article_2}")

        labels = [label for label, _ in fragments]
        self.assertGreaterEqual(len(labels), 2)
        self.assertTrue(any("ARTICULO 1" in label for label in labels))
        self.assertTrue(any("ARTICULO 2" in label for label in labels))


if __name__ == '__main__':
    unittest.main(verbosity=2)
