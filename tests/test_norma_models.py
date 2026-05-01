"""Tests for norma_models: dataclasses, regex parsing, and normalization."""

import unittest

from src.data_preparation.norma_models import (
    GOC_CODE_PATTERN,
    HEADER_TYPE_PATTERN,
    NormaIdentity,
    Norma,
    Gaceta,
    ProcessingResult,
    extract_header_identity,
    normalize_numero,
    normalize_tipo,
    parse_norma_name,
)


class TestParseNormaName(unittest.TestCase):
    """Validate regex parsing of gaceta_normas metadata strings."""

    def test_resolucion_basic(self):
        r = parse_norma_name("Resolución 41 de 2026 de Ministerio de Finanzas y Precios")
        self.assertIsNotNone(r)
        self.assertEqual(normalize_tipo(r.tipo), "resolucion")
        self.assertEqual(r.numero, "41")
        self.assertEqual(r.year, 2026)
        self.assertEqual(r.organismo_emisor, "Ministerio de Finanzas y Precios")

    def test_decreto_ley(self):
        r = parse_norma_name("Decreto Ley 114 de 2025 de Consejo de Estado")
        self.assertIsNotNone(r)
        self.assertEqual(normalize_tipo(r.tipo), "decreto ley")
        self.assertEqual(r.numero, "114")
        self.assertEqual(r.year, 2025)

    def test_decreto_presidencial(self):
        r = parse_norma_name("Decreto Presidencial 1177 de 2026 de Presidente de la República")
        self.assertIsNotNone(r)
        self.assertEqual(normalize_tipo(r.tipo), "decreto presidencial")
        self.assertEqual(r.numero, "1177")

    def test_proclama_sin_numero(self):
        r = parse_norma_name("Proclama S/N de 2026 de Consejo de Estado")
        self.assertIsNotNone(r)
        self.assertEqual(normalize_tipo(r.tipo), "proclama")
        self.assertEqual(r.numero, "S/N")

    def test_acuerdo_x_prefix(self):
        r = parse_norma_name("Acuerdo X-144 de 2025 de Asamblea Nacional del Poder Popular")
        self.assertIsNotNone(r)
        self.assertEqual(r.numero, "X-144")

    def test_acuerdo_large_number(self):
        r = parse_norma_name("Acuerdo 10249 de 2025 de Consejo de Ministros")
        self.assertIsNotNone(r)
        self.assertEqual(r.numero, "10249")

    def test_nota_presentacion(self):
        r = parse_norma_name(
            "Nota de Presentación de Cartas Credenciales s/n de 2025 "
            "de Presidente de la República"
        )
        self.assertIsNotNone(r)
        self.assertIn("nota", normalize_tipo(r.tipo))
        self.assertEqual(r.numero, "s/n")

    def test_instruccion_conjunta(self):
        r = parse_norma_name(
            "Instrucción Conjunta 1 de 2023 de Ministerio de Justicia, "
            "Ministerio de Trabajo y Seguridad Social"
        )
        self.assertIsNotNone(r)
        self.assertEqual(normalize_tipo(r.tipo), "instruccion conjunta")
        self.assertEqual(r.numero, "1")

    def test_resolucion_conjunta(self):
        r = parse_norma_name("Resolución Conjunta 1 de 2024 de Ministerio de Finanzas y Precios")
        self.assertIsNotNone(r)
        self.assertEqual(normalize_tipo(r.tipo), "resolucion conjunta")

    def test_constitucion_no_numero(self):
        r = parse_norma_name("Constitución de 2019 de Asamblea Nacional del Poder Popular")
        self.assertIsNotNone(r)
        self.assertEqual(normalize_tipo(r.tipo), "constitucion")
        self.assertEqual(r.numero, "s/n")
        self.assertEqual(r.year, 2019)

    def test_constitucion_extra_spaces(self):
        r = parse_norma_name("Constitución   de 2019 de Asamblea Nacional del Poder Popular")
        self.assertIsNotNone(r)
        self.assertEqual(r.year, 2019)

    def test_ley_basic(self):
        r = parse_norma_name("Ley 177 de 2025 de Asamblea Nacional del Poder Popular")
        self.assertIsNotNone(r)
        self.assertEqual(normalize_tipo(r.tipo), "ley")
        self.assertEqual(r.numero, "177")

    def test_decreto_basic(self):
        r = parse_norma_name("Decreto 141 de 2025 de Consejo de Ministros")
        self.assertIsNotNone(r)
        self.assertEqual(normalize_tipo(r.tipo), "decreto")
        self.assertEqual(r.numero, "141")

    def test_resolucion_v_prefix(self):
        r = parse_norma_name(
            "Resolución V-03 de 2026 de Ministerio de la Construcción"
        )
        self.assertIsNotNone(r)
        self.assertEqual(r.numero, "V-03")

    def test_empty_string_returns_none(self):
        self.assertIsNone(parse_norma_name(""))

    def test_unparseable_returns_none(self):
        self.assertIsNone(parse_norma_name("some random text"))

    def test_none_input_returns_none(self):
        self.assertIsNone(parse_norma_name(None))

    def test_no_accent_resolucion(self):
        r = parse_norma_name("Resolucion 41 de 2026 de Ministerio de Finanzas y Precios")
        self.assertIsNotNone(r)
        self.assertEqual(r.numero, "41")


class TestNormalizeTipo(unittest.TestCase):

    def test_decreto_ley_variants(self):
        self.assertEqual(normalize_tipo("Decreto-Ley"), "decreto ley")
        self.assertEqual(normalize_tipo("Decreto Ley"), "decreto ley")
        self.assertEqual(normalize_tipo("DECRETO-LEY"), "decreto ley")
        self.assertEqual(normalize_tipo("DECRETO LEY"), "decreto ley")

    def test_resolucion_accent(self):
        self.assertEqual(normalize_tipo("Resolución"), "resolucion")
        self.assertEqual(normalize_tipo("RESOLUCIÓN"), "resolucion")
        self.assertEqual(normalize_tipo("Resolucion"), "resolucion")


class TestNormalizeNumero(unittest.TestCase):

    def test_strip_year_suffix(self):
        self.assertEqual(normalize_numero("8/2026"), "8")
        self.assertEqual(normalize_numero("41/2026"), "41")

    def test_strip_leading_zeros(self):
        self.assertEqual(normalize_numero("08"), "8")
        self.assertEqual(normalize_numero("003"), "3")

    def test_preserve_non_numeric(self):
        self.assertEqual(normalize_numero("X-144"), "x-144")
        self.assertEqual(normalize_numero("V-03"), "v-03")
        self.assertEqual(normalize_numero("s/n"), "s/n")


class TestGOCCodePattern(unittest.TestCase):

    def test_ordinaria(self):
        m = GOC_CODE_PATTERN.search("texto GOC-2026-215-O24 mas texto")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "GOC-2026-215-O24")

    def test_extraordinaria(self):
        m = GOC_CODE_PATTERN.search("GOC-2026-179-EX30")
        self.assertIsNotNone(m)

    def test_extraordinaria_especial(self):
        m = GOC_CODE_PATTERN.search("GOC-2022-296-EXE2")
        self.assertIsNotNone(m)

    def test_multiple_goc_codes_in_text(self):
        text = "GOC-2022-100-EX9 RESOLUCION 1 GOC-2022-101-EX10 RESOLUCION 2"
        matches = GOC_CODE_PATTERN.findall(text)
        self.assertEqual(len(matches), 2)

    def test_case_insensitive(self):
        m = GOC_CODE_PATTERN.search("goc-2026-215-o24")
        self.assertIsNotNone(m)

    def test_no_match(self):
        m = GOC_CODE_PATTERN.search("texto sin codigo")
        self.assertIsNone(m)


class TestExtractHeaderIdentity(unittest.TestCase):

    def test_resolucion_with_year(self):
        r = extract_header_identity("GOC-2026-179-EX30 RESOLUCION 41/2026 POR CUANTO")
        self.assertIsNotNone(r)
        tipo, numero = r
        self.assertEqual(normalize_tipo(tipo), "resolucion")
        self.assertEqual(numero, "41")

    def test_decreto_ley(self):
        r = extract_header_identity("GOC-2026-215-O24 DECRETO-LEY 114 DE LA ASOCIACION")
        self.assertIsNotNone(r)
        tipo, numero = r
        self.assertEqual(normalize_tipo(tipo), "decreto ley")
        self.assertEqual(numero, "114")

    def test_acuerdo(self):
        r = extract_header_identity("GOC-2026-100-O15 ACUERDO X-144 DE 2025")
        self.assertIsNotNone(r)
        tipo, numero = r
        self.assertEqual(normalize_tipo(tipo), "acuerdo")
        self.assertEqual(numero, "X-144")

    def test_no_header(self):
        r = extract_header_identity("texto sin encabezado de norma conocido")
        self.assertIsNone(r)

    def test_empty_text(self):
        r = extract_header_identity("")
        self.assertIsNone(r)

    def test_none_text(self):
        r = extract_header_identity(None)
        self.assertIsNone(r)


class TestNormaIdentity(unittest.TestCase):

    def test_norma_id_basic(self):
        ni = NormaIdentity("Resolución", "41", 2026, "Ministerio de Finanzas y Precios")
        self.assertIn("resolucion", ni.norma_id)
        self.assertIn("41", ni.norma_id)
        self.assertIn("2026", ni.norma_id)

    def test_norma_id_decreto_ley(self):
        ni = NormaIdentity("Decreto Ley", "114", 2025, "Consejo de Estado")
        self.assertTrue(ni.norma_id.startswith("decreto_ley_"))

    def test_to_dict_roundtrip(self):
        ni = NormaIdentity("Ley", "177", 2025, "Asamblea Nacional", raw_string="Ley 177 de 2025")
        d = ni.to_dict()
        self.assertEqual(d['tipo'], "Ley")
        self.assertEqual(d['numero'], "177")
        self.assertEqual(d['year'], 2025)
        self.assertIn('norma_id', d)

    def test_norma_id_truncates_long_organismo(self):
        ni = NormaIdentity("Ley", "1", 2025, "A" * 100)
        # organismo part should be truncated to 50 chars
        parts = ni.norma_id.split("_")
        # The organismo part starts after ley_1_2025_
        organismo_part = ni.norma_id.split(f"_1_2025_")[1]
        self.assertLessEqual(len(organismo_part), 50)


class TestNormaDataclass(unittest.TestCase):

    def test_to_dict(self):
        ni = NormaIdentity("Resolución", "41", 2026, "MFP")
        norma = Norma(
            identity=ni,
            goc_code="GOC-2026-179-EX30",
            raw_text="texto de la norma...",
            page_range=(1, 5),
            ordinal_position=0,
            match_confidence="high",
        )
        d = norma.to_dict()
        self.assertEqual(d['goc_code'], "GOC-2026-179-EX30")
        self.assertEqual(d['page_range'], [1, 5])
        self.assertIsInstance(d['identity'], dict)


class TestGacetaDataclass(unittest.TestCase):

    def test_grouping_key_with_checksum(self):
        g = Gaceta("24", "03 Marzo, 2026", "Ordinaria", checksum="abc123")
        self.assertEqual(g.grouping_key, "abc123")

    def test_grouping_key_without_checksum(self):
        g = Gaceta("24", "03 Marzo, 2026", "Ordinaria")
        self.assertEqual(g.grouping_key, "24_03 Marzo, 2026")

    def test_to_dict(self):
        g = Gaceta("24", "03 Marzo, 2026", "Ordinaria")
        d = g.to_dict()
        self.assertEqual(d['norma_count'], 0)
        self.assertIsInstance(d['normas'], list)


class TestProcessingResult(unittest.TestCase):

    def test_to_dict_structure(self):
        pr = ProcessingResult(
            total_chunks_processed=100,
            total_normas_extracted=5,
        )
        d = pr.to_dict()
        self.assertIn('stats', d)
        self.assertIn('gacetas', d)
        self.assertEqual(d['stats']['total_chunks_processed'], 100)


if __name__ == '__main__':
    unittest.main()
