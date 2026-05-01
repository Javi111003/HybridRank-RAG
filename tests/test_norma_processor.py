"""Tests for norma_processor pipeline functions."""

import sqlite3
import tempfile
import os
import unittest

from src.data_preparation.norma_models import NormaIdentity, parse_norma_name
from src.data_preparation.norma_processor import (
    _offset_to_page,
    _parse_gaceta_fecha,
    concatenate_gaceta_text,
    detect_cross_gaceta_duplicates,
    group_chunks_by_gaceta,
    match_segments_to_normas,
    process_all_normas,
    process_single_gaceta,
    save_sqlite_output,
    segment_by_goc_codes,
)


def _make_chunk(content, page_number=1, gaceta_numero="24", gaceta_fecha="03 Marzo, 2026",
                gaceta_checksum="abc123", gaceta_normas=None, **extra_meta):
    """Helper to build a chunk dict matching the cleaned_elements.json structure."""
    meta = {
        "source": "test.pdf",
        "page_number": page_number,
        "type": "CompositeElement",
        "chunk_id": f"test-{id(content)}",
        "document_type": "gaceta",
        "gaceta_tipo_edicion": "Ordinaria",
        "gaceta_fecha": gaceta_fecha,
        "gaceta_numero": gaceta_numero,
        "gaceta_normas": gaceta_normas or [],
        "gaceta_pdf_url": "https://example.com/test.pdf",
        "gaceta_checksum": gaceta_checksum,
        "gaceta_relative_path": "full/test.pdf",
    }
    meta.update(extra_meta)
    return {"content": content, "cleaned_content": content.lower(), "metadata": meta}


class TestGroupChunksByGaceta(unittest.TestCase):

    def test_groups_by_checksum(self):
        chunks = [
            _make_chunk("texto 1", gaceta_checksum="aaa"),
            _make_chunk("texto 2", gaceta_checksum="aaa"),
            _make_chunk("texto 3", gaceta_checksum="bbb"),
        ]
        groups = group_chunks_by_gaceta(chunks)
        self.assertEqual(len(groups), 2)
        self.assertEqual(len(groups["aaa"]), 2)
        self.assertEqual(len(groups["bbb"]), 1)

    def test_fallback_to_numero_fecha(self):
        chunks = [
            _make_chunk("t1", gaceta_checksum="", gaceta_numero="10", gaceta_fecha="01 Enero, 2026"),
            _make_chunk("t2", gaceta_checksum="", gaceta_numero="10", gaceta_fecha="01 Enero, 2026"),
        ]
        groups = group_chunks_by_gaceta(chunks)
        self.assertIn("10_01 Enero, 2026", groups)

    def test_skips_chunks_without_metadata(self):
        chunks = [
            {"content": "no meta", "metadata": {}},
            _make_chunk("good", gaceta_checksum="x"),
        ]
        groups = group_chunks_by_gaceta(chunks)
        self.assertEqual(len(groups), 1)


class TestConcatenateGacetaText(unittest.TestCase):

    def test_sorts_by_page_number(self):
        chunks = [
            _make_chunk("page 3", page_number=3),
            _make_chunk("page 1", page_number=1),
            _make_chunk("page 2", page_number=2),
        ]
        text, boundaries = concatenate_gaceta_text(chunks)
        self.assertTrue(text.startswith("page 1"))
        self.assertIn("page 2", text)
        self.assertTrue(text.endswith("page 3"))

    def test_uses_content_not_cleaned(self):
        chunks = [_make_chunk("ORIGINAL TEXT", page_number=1)]
        text, _ = concatenate_gaceta_text(chunks)
        self.assertIn("ORIGINAL TEXT", text)

    def test_handles_none_page_number(self):
        chunks = [
            _make_chunk("no page", page_number=None),
            _make_chunk("page 1", page_number=1),
        ]
        text, _ = concatenate_gaceta_text(chunks)
        # None page defaults to 0, so it comes first
        self.assertTrue(text.startswith("no page"))

    def test_page_boundaries(self):
        chunks = [
            _make_chunk("aaaa", page_number=1),
            _make_chunk("bbbb", page_number=2),
        ]
        text, boundaries = concatenate_gaceta_text(chunks)
        self.assertEqual(len(boundaries), 2)
        self.assertEqual(boundaries[0], (0, 1))
        # Second offset = len("aaaa") + 1 (separator)
        self.assertEqual(boundaries[1], (5, 2))


class TestOffsetToPage(unittest.TestCase):

    def test_basic(self):
        boundaries = [(0, 1), (100, 2), (200, 3)]
        self.assertEqual(_offset_to_page(50, boundaries), 1)
        self.assertEqual(_offset_to_page(150, boundaries), 2)
        self.assertEqual(_offset_to_page(250, boundaries), 3)

    def test_empty_boundaries(self):
        self.assertEqual(_offset_to_page(10, []), 0)

    def test_exact_boundary(self):
        boundaries = [(0, 1), (100, 2)]
        self.assertEqual(_offset_to_page(100, boundaries), 2)


class TestSegmentByGocCodes(unittest.TestCase):

    def test_two_goc_codes(self):
        text = "SUMARIO contenido\nGOC-2026-100-O24 RESOLUCION 1/2026 texto norm1\nGOC-2026-101-O24 RESOLUCION 2/2026 texto norm2"
        boundaries = [(0, 1)]
        sumario, segments = segment_by_goc_codes(text, boundaries)
        self.assertEqual(len(segments), 2)
        self.assertIn("SUMARIO", sumario)
        self.assertEqual(segments[0]['goc_code'], "GOC-2026-100-O24")
        self.assertEqual(segments[1]['goc_code'], "GOC-2026-101-O24")
        self.assertIn("RESOLUCION 1/2026", segments[0]['text'])
        self.assertIn("RESOLUCION 2/2026", segments[1]['text'])

    def test_no_goc_codes(self):
        text = "texto sin codigos GOC"
        sumario, segments = segment_by_goc_codes(text, [(0, 1)])
        self.assertEqual(sumario, text)
        self.assertEqual(segments, [])

    def test_single_goc_code(self):
        text = "SUMARIO\nGOC-2026-179-EX30 RESOLUCION 41/2026 cuerpo"
        sumario, segments = segment_by_goc_codes(text, [(0, 1)])
        self.assertEqual(len(segments), 1)
        self.assertIn("SUMARIO", sumario)

    def test_page_range_computed(self):
        # Chunk on page 1, then chunk on page 5
        text = ("a" * 100) + "\nGOC-2026-100-O24 norm text" + ("b" * 50) + "\nGOC-2026-101-O24 norm2"
        boundaries = [(0, 1), (50, 3), (100, 5)]
        _, segments = segment_by_goc_codes(text, boundaries)
        self.assertEqual(len(segments), 2)
        # First segment starts after position 101, page should be 5
        self.assertEqual(segments[0]['page_range'][0], 5)


class TestMatchSegmentsToNormas(unittest.TestCase):

    def test_high_confidence_match(self):
        segments = [
            {
                'goc_code': 'GOC-2026-100-O24',
                'text': 'GOC-2026-100-O24 RESOLUCION 41/2026 POR CUANTO...',
                'page_range': (1, 5),
                'ordinal_position': 0,
            },
        ]
        parsed = [
            NormaIdentity("Resolución", "41", 2026, "Ministerio de Finanzas y Precios",
                          raw_string="Resolución 41 de 2026 de Ministerio de Finanzas y Precios"),
        ]
        matched, unmatched = match_segments_to_normas(segments, parsed)
        self.assertEqual(len(matched), 1)
        self.assertEqual(len(unmatched), 0)
        self.assertEqual(matched[0].match_confidence, 'high')
        self.assertEqual(matched[0].identity.numero, "41")

    def test_sn_positional_match(self):
        segments = [
            {
                'goc_code': 'GOC-2026-100-O24',
                'text': 'GOC-2026-100-O24 PROCLAMA algo ceremonial...',
                'page_range': (1, 2),
                'ordinal_position': 0,
            },
        ]
        parsed = [
            NormaIdentity("Proclama", "S/N", 2026, "Consejo de Estado",
                          raw_string="Proclama S/N de 2026 de Consejo de Estado"),
        ]
        matched, unmatched = match_segments_to_normas(segments, parsed)
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0].match_confidence, 'medium')

    def test_unmatched_goes_to_bucket(self):
        segments = [
            {
                'goc_code': 'GOC-2026-100-O24',
                'text': 'GOC-2026-100-O24 SOMETHING UNKNOWN',
                'page_range': (1, 1),
                'ordinal_position': 0,
            },
        ]
        parsed = []  # no metadata to match
        matched, unmatched = match_segments_to_normas(segments, parsed)
        self.assertEqual(len(matched), 0)
        self.assertEqual(len(unmatched), 1)

    def test_multiple_normas_matched(self):
        segments = [
            {
                'goc_code': 'GOC-2026-100-O24',
                'text': 'GOC-2026-100-O24 DECRETO-LEY 114 DE LA ASOCIACION...',
                'page_range': (1, 10),
                'ordinal_position': 0,
            },
            {
                'goc_code': 'GOC-2026-101-O24',
                'text': 'GOC-2026-101-O24 RESOLUCION 8/2026 POR CUANTO...',
                'page_range': (11, 15),
                'ordinal_position': 1,
            },
        ]
        parsed = [
            NormaIdentity("Decreto Ley", "114", 2025, "Consejo de Estado"),
            NormaIdentity("Resolución", "8", 2026, "Ministerio de Economía"),
        ]
        matched, unmatched = match_segments_to_normas(segments, parsed)
        self.assertEqual(len(matched), 2)
        self.assertEqual(len(unmatched), 0)
        self.assertEqual(matched[0].identity.numero, "114")
        self.assertEqual(matched[1].identity.numero, "8")


class TestProcessSingleGaceta(unittest.TestCase):

    def test_single_norma_with_goc(self):
        chunks = [
            _make_chunk(
                "SUMARIO\nResolución 41...\n",
                page_number=1,
                gaceta_normas=["Resolución 41 de 2026 de Ministerio de Finanzas y Precios"],
            ),
            _make_chunk(
                "GOC-2026-179-EX30 RESOLUCION 41/2026 POR CUANTO: El Ministerio...",
                page_number=2,
                gaceta_normas=["Resolución 41 de 2026 de Ministerio de Finanzas y Precios"],
            ),
        ]
        gaceta = process_single_gaceta("test_key", chunks)
        self.assertEqual(len(gaceta.normas), 1)
        self.assertEqual(gaceta.normas[0].match_confidence, 'high')
        self.assertIn("SUMARIO", gaceta.sumario_text)

    def test_single_norma_no_goc(self):
        chunks = [
            _make_chunk(
                "RESOLUCION 41/2026 POR CUANTO: texto completo sin GOC...",
                page_number=1,
                gaceta_normas=["Resolución 41 de 2026 de MFP"],
            ),
        ]
        gaceta = process_single_gaceta("test_key", chunks)
        self.assertEqual(len(gaceta.normas), 1)
        self.assertEqual(gaceta.normas[0].match_confidence, 'medium')

    def test_multiple_normas(self):
        chunks = [
            _make_chunk(
                "SUMARIO...",
                page_number=1,
                gaceta_normas=[
                    "Decreto Ley 114 de 2025 de Consejo de Estado",
                    "Resolución 8 de 2026 de Ministerio de Economía y Planificación",
                ],
            ),
            _make_chunk(
                "GOC-2026-215-O24 DECRETO-LEY 114 DE LA ASOCIACION ENTRE ENTIDADES "
                "POR CUANTO: largo texto...",
                page_number=2,
                gaceta_normas=[
                    "Decreto Ley 114 de 2025 de Consejo de Estado",
                    "Resolución 8 de 2026 de Ministerio de Economía y Planificación",
                ],
            ),
            _make_chunk(
                "GOC-2026-216-O24 RESOLUCION 8/2026 POR CUANTO: El Ministerio...",
                page_number=10,
                gaceta_normas=[
                    "Decreto Ley 114 de 2025 de Consejo de Estado",
                    "Resolución 8 de 2026 de Ministerio de Economía y Planificación",
                ],
            ),
        ]
        gaceta = process_single_gaceta("test_key", chunks)
        self.assertEqual(len(gaceta.normas), 2)
        tipos = {n.identity.tipo for n in gaceta.normas}
        self.assertTrue(any("ecreto" in t for t in tipos))
        self.assertTrue(any("esoluci" in t for t in tipos))


class TestDetectCrossGacetaDuplicates(unittest.TestCase):

    def test_detects_duplicate(self):
        from src.data_preparation.norma_models import Gaceta, Norma
        ni = NormaIdentity("Resolución", "13", 2025, "Ministerio del Interior")
        gaceta1 = Gaceta("34", "09 Julio, 2025", "Extraordinaria", checksum="aaa")
        gaceta1.normas = [Norma(identity=ni, goc_code="GOC-2025-100-EX34", raw_text="t1")]
        gaceta2 = Gaceta("50", "20 Mayo, 2025", "Ordinaria", checksum="bbb")
        gaceta2.normas = [Norma(identity=ni, goc_code="GOC-2025-50-O50", raw_text="t2")]

        dups = detect_cross_gaceta_duplicates([gaceta1, gaceta2])
        self.assertEqual(len(dups), 1)
        self.assertEqual(dups[0]['kept']['gaceta_checksum'], "aaa")  # July > May

    def test_no_duplicates(self):
        from src.data_preparation.norma_models import Gaceta, Norma
        ni1 = NormaIdentity("Resolución", "1", 2025, "MFP")
        ni2 = NormaIdentity("Resolución", "2", 2025, "MFP")
        g1 = Gaceta("1", "01 Enero, 2025", "Ordinaria")
        g1.normas = [Norma(identity=ni1, goc_code="GOC-1", raw_text="t")]
        g2 = Gaceta("2", "02 Febrero, 2025", "Ordinaria")
        g2.normas = [Norma(identity=ni2, goc_code="GOC-2", raw_text="t")]
        dups = detect_cross_gaceta_duplicates([g1, g2])
        self.assertEqual(len(dups), 0)


class TestParseGacetaFecha(unittest.TestCase):

    def test_standard_format(self):
        r = _parse_gaceta_fecha("03 Marzo, 2026")
        self.assertEqual(r, (2026, 3, 3))

    def test_single_digit_day(self):
        r = _parse_gaceta_fecha("9 Enero, 2025")
        self.assertEqual(r, (2025, 1, 9))

    def test_invalid_returns_none(self):
        self.assertIsNone(_parse_gaceta_fecha("invalid"))


class TestEndToEnd(unittest.TestCase):

    def test_small_dataset(self):
        """Process a minimal synthetic dataset end-to-end."""
        norma_list = [
            "Resolución 41 de 2026 de Ministerio de Finanzas y Precios",
            "Resolución 42 de 2026 de Ministerio de Finanzas y Precios",
        ]
        chunks = [
            _make_chunk(
                "SUMARIO\nResolución 41... Resolución 42...\n",
                page_number=1, gaceta_normas=norma_list,
            ),
            _make_chunk(
                "GOC-2026-179-EX30 RESOLUCION 41/2026 POR CUANTO: texto de la primera",
                page_number=2, gaceta_normas=norma_list,
            ),
            _make_chunk(
                "GOC-2026-180-EX30 RESOLUCION 42/2026 POR CUANTO: texto de la segunda",
                page_number=5, gaceta_normas=norma_list,
            ),
        ]
        result = process_all_normas(chunks)
        self.assertEqual(len(result.gacetas), 1)
        gaceta = result.gacetas[0]
        self.assertEqual(len(gaceta.normas), 2)
        self.assertEqual(result.total_normas_extracted, 2)
        self.assertEqual(result.total_unmatched_segments, 0)

    def test_sqlite_output(self):
        """Verify SQLite output is valid."""
        norma_list = ["Ley 1 de 2025 de Asamblea Nacional del Poder Popular"]
        chunks = [
            _make_chunk(
                "GOC-2025-100-O10 LEY 1 DE 2025 POR CUANTO...",
                page_number=1, gaceta_normas=norma_list,
            ),
        ]
        result = process_all_normas(chunks)

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            save_sqlite_output(result, db_path)

            conn = sqlite3.connect(db_path)
            gaceta_count = conn.execute("SELECT COUNT(*) FROM gacetas").fetchone()[0]
            norma_count = conn.execute("SELECT COUNT(*) FROM normas").fetchone()[0]
            conn.close()

            self.assertEqual(gaceta_count, 1)
            self.assertEqual(norma_count, 1)


if __name__ == '__main__':
    unittest.main()
