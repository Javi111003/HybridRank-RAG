"""Tests for norma_linker: reference extraction, classification, resolution, and integration."""

import sqlite3
import tempfile
import os
import unittest

from src.data_preparation.norma_models import (
    NormaReference,
    NormaRelationship,
    LinkingResult,
    normalize_tipo,
    normalize_numero,
)
from src.data_preparation.norma_linker import (
    REFERENCE_PATTERN,
    LOOSE_REFERENCE_PATTERN,
    DISPOSICIONES_PATTERN,
    DEROGA_VERBS,
    MODIFICA_VERBS,
    COMPLEMENTA_VERBS,
    extract_disposiciones_section,
    extract_references,
    extract_loose_references,
    classify_relation_type,
    normalize_organismo,
    fuzzy_organismo_match,
    resolve_references,
    resolve_loose_references,
    build_norma_index,
    link_all_normas,
    save_relationships_sqlite,
    save_linking_report,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_in_memory_db(normas=None):
    """Create an in-memory SQLite DB with Phase 1 schema and optional norma data."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE gacetas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            numero TEXT NOT NULL,
            fecha TEXT NOT NULL,
            tipo_edicion TEXT NOT NULL,
            pdf_url TEXT,
            checksum TEXT UNIQUE,
            sumario_text TEXT,
            norma_count INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
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
            gaceta_id INTEGER REFERENCES gacetas(id),
            raw_metadata_string TEXT
        )
    """)
    conn.execute("CREATE INDEX idx_normas_norma_id ON normas(norma_id)")

    if normas:
        # Insert a default gaceta first
        conn.execute(
            "INSERT INTO gacetas (numero, fecha, tipo_edicion, checksum) VALUES (?, ?, ?, ?)",
            ("30", "03 Marzo, 2026", "Extraordinaria", "test_checksum"),
        )
        for n in normas:
            conn.execute(
                "INSERT INTO normas (norma_id, tipo, numero, year, organismo_emisor, "
                "goc_code, raw_text, gaceta_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (n['norma_id'], n['tipo'], n['numero'], n['year'],
                 n['organismo_emisor'], n['goc_code'], n['raw_text'], 1),
            )
    conn.commit()
    return conn


def _sample_normas():
    """Standard set of normas for resolution tests."""
    return [
        {
            'norma_id': 'resolucion_41_2026_ministerio_de_finanzas_y_precios',
            'tipo': 'Resolución', 'numero': '41', 'year': 2026,
            'organismo_emisor': 'Ministerio de Finanzas y Precios',
            'goc_code': 'GOC-2026-179-EX30',
            'raw_text': 'RESOLUCION 41/2026 POR CUANTO...',
        },
        {
            'norma_id': 'decreto_ley_114_2025_consejo_de_estado',
            'tipo': 'Decreto Ley', 'numero': '114', 'year': 2025,
            'organismo_emisor': 'Consejo de Estado',
            'goc_code': 'GOC-2025-215-O24',
            'raw_text': 'DECRETO-LEY 114 DE LA ASOCIACION...',
        },
        {
            'norma_id': 'resolucion_8_2026_ministerio_de_economia_y_planificacion',
            'tipo': 'Resolución', 'numero': '8', 'year': 2026,
            'organismo_emisor': 'Ministerio de Economía y Planificación',
            'goc_code': 'GOC-2026-216-O24',
            'raw_text': 'RESOLUCION 8/2026 POR CUANTO...',
        },
        {
            'norma_id': 'ley_177_2025_asamblea_nacional_del_poder_popular',
            'tipo': 'Ley', 'numero': '177', 'year': 2025,
            'organismo_emisor': 'Asamblea Nacional del Poder Popular',
            'goc_code': 'GOC-2025-100-O10',
            'raw_text': 'LEY 177 DE 2025...',
        },
    ]


# ── Reference Extraction ────────────────────────────────────────────────────

class TestReferencePattern(unittest.TestCase):

    def test_resolucion_with_organismo(self):
        text = "se deroga la Resolución No. 41 de 2026 del Ministerio de Finanzas y Precios"
        m = REFERENCE_PATTERN.search(text)
        self.assertIsNotNone(m)
        self.assertIn("esoluci", m.group('tipo'))
        self.assertEqual(m.group('numero'), "41")
        self.assertEqual(m.group('year'), "2026")
        self.assertIsNotNone(m.group('organismo'))

    def test_decreto_ley(self):
        text = "lo dispuesto en el Decreto-Ley 114 de 2025"
        m = REFERENCE_PATTERN.search(text)
        self.assertIsNotNone(m)
        self.assertEqual(m.group('numero'), "114")
        self.assertEqual(m.group('year'), "2025")

    def test_without_organismo(self):
        text = "conforme a la Resolución 8 de 2026."
        m = REFERENCE_PATTERN.search(text)
        self.assertIsNotNone(m)
        self.assertEqual(m.group('numero'), "8")
        self.assertIsNone(m.group('organismo'))

    def test_ley_basic(self):
        text = "según establece la Ley 177 de 2025 de Asamblea Nacional"
        m = REFERENCE_PATTERN.search(text)
        self.assertIsNotNone(m)
        self.assertEqual(m.group('numero'), "177")

    def test_with_date(self):
        text = "la Resolución No. 41 de 3 de marzo de 2026 del Ministerio de Finanzas"
        m = REFERENCE_PATTERN.search(text)
        self.assertIsNotNone(m)
        self.assertEqual(m.group('numero'), "41")
        self.assertEqual(m.group('year'), "2026")

    def test_multiple_references(self):
        text = (
            "Se derogan la Resolución 41 de 2026 del Ministerio de Finanzas y Precios "
            "y el Decreto-Ley 114 de 2025 del Consejo de Estado."
        )
        matches = list(REFERENCE_PATTERN.finditer(text))
        self.assertEqual(len(matches), 2)

    def test_no_match(self):
        text = "texto sin referencias a normas legales"
        m = REFERENCE_PATTERN.search(text)
        self.assertIsNone(m)

    def test_acuerdo(self):
        text = "el Acuerdo 10249 de 2025 del Consejo de Ministros"
        m = REFERENCE_PATTERN.search(text)
        self.assertIsNotNone(m)
        self.assertEqual(m.group('numero'), "10249")

    def test_resolucion_conjunta(self):
        text = "la Resolución Conjunta 1 de 2024 del Ministerio de Finanzas"
        m = REFERENCE_PATTERN.search(text)
        self.assertIsNotNone(m)
        self.assertIn("Conjunta", m.group('tipo'))

    def test_numero_with_year_suffix(self):
        text = "la Resolución 41/2026 de 2026 del Ministerio de Finanzas"
        m = REFERENCE_PATTERN.search(text)
        self.assertIsNotNone(m)


# ── DISPOSICIONES Section Extraction ─────────────────────────────────────────

class TestDisposicionesExtraction(unittest.TestCase):

    def test_finds_disposiciones_finales(self):
        text = (
            "ARTICULO 1: Algo.\n"
            "DISPOSICIONES FINALES\n"
            "PRIMERA: Se deroga la Resolución 41 de 2026.\n"
            "SEGUNDA: La presente entra en vigor."
        )
        section = extract_disposiciones_section(text)
        self.assertIn("Se deroga", section)
        self.assertIn("PRIMERA", section)

    def test_disposicion_final_singular(self):
        text = (
            "ARTICULO 10: Texto.\n"
            "DISPOSICION FINAL\n"
            "Se deroga la Resolución 8 de 2026."
        )
        section = extract_disposiciones_section(text)
        self.assertIn("Se deroga", section)

    def test_no_disposiciones_returns_none(self):
        text = "ARTICULO 1: Algo.\nARTICULO 2: Otra cosa."
        section = extract_disposiciones_section(text)
        self.assertIsNone(section)

    def test_empty_text(self):
        self.assertIsNone(extract_disposiciones_section(""))
        self.assertIsNone(extract_disposiciones_section(None))


# ── Relation Type Classification ─────────────────────────────────────────────

class TestClassifyRelationType(unittest.TestCase):

    def test_deroga(self):
        self.assertEqual(
            classify_relation_type("se deroga la Resolución 41 de 2026"),
            'deroga',
        )

    def test_deroga_queda_derogada(self):
        self.assertEqual(
            classify_relation_type("queda derogada la Resolución 41 de 2026"),
            'deroga',
        )

    def test_dejar_sin_efecto(self):
        self.assertEqual(
            classify_relation_type("dejar sin efecto la Resolución 8 de 2026"),
            'deroga',
        )

    def test_modifica(self):
        self.assertEqual(
            classify_relation_type("se modifica el artículo 5 de la Ley 177 de 2025"),
            'modifica',
        )

    def test_modifica_se_sustituye(self):
        self.assertEqual(
            classify_relation_type("se sustituye el inciso a) del artículo 3"),
            'modifica',
        )

    def test_modifica_se_adiciona(self):
        self.assertEqual(
            classify_relation_type("se adicionan los artículos 15 y 16"),
            'modifica',
        )

    def test_complementa(self):
        self.assertEqual(
            classify_relation_type("en lo que complementa la Resolución 41 de 2026"),
            'complementa',
        )

    def test_menciona_default(self):
        self.assertEqual(
            classify_relation_type("según lo dispuesto en la Resolución 41 de 2026"),
            'menciona',
        )


# ── Organismo Matching ───────────────────────────────────────────────────────

class TestNormalizeOrganismo(unittest.TestCase):

    def test_basic(self):
        result = normalize_organismo("Ministerio de Finanzas y Precios")
        self.assertIn("ministerio", result)
        self.assertIn("finanzas", result)
        self.assertIn("precios", result)

    def test_strips_accents(self):
        result = normalize_organismo("Ministerio de Economía y Planificación")
        self.assertIn("economia", result)
        self.assertIn("planificacion", result)

    def test_lowercase(self):
        result = normalize_organismo("CONSEJO DE ESTADO")
        self.assertEqual(result, normalize_organismo("Consejo de Estado"))


class TestFuzzyOrganismoMatch(unittest.TestCase):

    def test_ministra_vs_ministerio(self):
        self.assertTrue(
            fuzzy_organismo_match(
                "ministra de Finanzas y Precios",
                "Ministerio de Finanzas y Precios",
            )
        )

    def test_exact_match(self):
        self.assertTrue(
            fuzzy_organismo_match(
                "Ministerio de Finanzas y Precios",
                "Ministerio de Finanzas y Precios",
            )
        )

    def test_no_match(self):
        self.assertFalse(
            fuzzy_organismo_match(
                "Ministerio de Salud Pública",
                "Ministerio de Finanzas y Precios",
            )
        )

    def test_partial_overlap_below_threshold(self):
        self.assertFalse(
            fuzzy_organismo_match("Ministerio", "Asamblea Nacional del Poder Popular")
        )

    def test_empty_strings(self):
        self.assertFalse(fuzzy_organismo_match("", "Ministerio de Finanzas"))
        self.assertFalse(fuzzy_organismo_match("Ministerio", ""))


# ── Reference Extraction (full function) ─────────────────────────────────────

class TestExtractReferences(unittest.TestCase):

    def test_single_reference_in_disposiciones(self):
        text = (
            "ARTICULO 1: Algo.\n"
            "DISPOSICIONES FINALES\n"
            "PRIMERA: Se deroga la Resolución No. 41 de 2026 del Ministerio de Finanzas y Precios.\n"
        )
        refs = extract_references("source_norma_id", text)
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].ref_numero, "41")
        self.assertEqual(refs[0].relation_type, 'deroga')
        self.assertEqual(refs[0].source_norma_id, "source_norma_id")

    def test_multiple_references(self):
        text = (
            "DISPOSICIONES FINALES\n"
            "PRIMERA: Se deroga la Resolución 41 de 2026 del Ministerio de Finanzas y Precios.\n"
            "SEGUNDA: Se modifica el Decreto-Ley 114 de 2025 del Consejo de Estado.\n"
        )
        refs = extract_references("src", text)
        self.assertEqual(len(refs), 2)
        tipos = {normalize_tipo(r.ref_tipo) for r in refs}
        self.assertIn("resolucion", tipos)
        self.assertIn("decreto ley", tipos)

    def test_no_disposiciones_falls_back_to_full_text(self):
        text = "Conforme a la Resolución 8 de 2026 del Ministerio de Economía."
        refs = extract_references("src", text)
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].relation_type, 'menciona')

    def test_no_references(self):
        text = "ARTICULO 1: Texto sin referencias.\nDISPOSICIONES FINALES\nPRIMERA: Vigencia."
        refs = extract_references("src", text)
        self.assertEqual(len(refs), 0)

    def test_self_reference_excluded(self):
        text = (
            "DISPOSICIONES FINALES\n"
            "PRIMERA: Se deroga la Resolución 41 de 2026 del Ministerio de Finanzas."
        )
        refs = extract_references(
            "resolucion_41_2026_ministerio_de_finanzas_y_precios", text
        )
        # Self-references should be excluded
        self.assertEqual(len(refs), 0)

    def test_deduplicates_same_reference(self):
        text = (
            "DISPOSICIONES FINALES\n"
            "PRIMERA: Se deroga la Resolución 41 de 2026 del MFP. "
            "SEGUNDA: La mencionada Resolución 41 de 2026 queda sin efecto."
        )
        refs = extract_references("other_norma", text)
        # Should deduplicate same tipo+numero+year
        unique_keys = {(r.ref_tipo, r.ref_numero, r.ref_year) for r in refs}
        self.assertEqual(len(refs), len(unique_keys))


# ── Resolution ───────────────────────────────────────────────────────────────

class TestResolveReferences(unittest.TestCase):

    def test_exact_match(self):
        conn = _make_in_memory_db(_sample_normas())
        index = build_norma_index(conn)
        ref = NormaReference(
            source_norma_id="other",
            ref_tipo="Resolución",
            ref_numero="41",
            ref_year=2026,
            ref_organismo="Ministerio de Finanzas y Precios",
        )
        resolved = resolve_references([ref], index)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].confidence, 'exact')
        self.assertIn("resolucion_41_2026", resolved[0].resolved_norma_id)
        conn.close()

    def test_fuzzy_organismo_resolution(self):
        conn = _make_in_memory_db(_sample_normas())
        index = build_norma_index(conn)
        ref = NormaReference(
            source_norma_id="other",
            ref_tipo="Resolución",
            ref_numero="41",
            ref_year=2026,
            ref_organismo="ministra de Finanzas y Precios",
        )
        resolved = resolve_references([ref], index)
        self.assertEqual(len(resolved), 1)
        self.assertIn(resolved[0].confidence, ('exact', 'fuzzy'))
        self.assertIsNotNone(resolved[0].resolved_norma_id)
        conn.close()

    def test_no_match_external(self):
        conn = _make_in_memory_db(_sample_normas())
        index = build_norma_index(conn)
        ref = NormaReference(
            source_norma_id="other",
            ref_tipo="Resolución",
            ref_numero="999",
            ref_year=2020,
        )
        resolved = resolve_references([ref], index)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].confidence, 'unresolved')
        self.assertIsNone(resolved[0].resolved_norma_id)
        conn.close()

    def test_ambiguous_no_organismo(self):
        """Two normas with same tipo+numero+year but different organismos, no ref_organismo."""
        normas = _sample_normas()
        normas.append({
            'norma_id': 'resolucion_41_2026_ministerio_de_salud_publica',
            'tipo': 'Resolución', 'numero': '41', 'year': 2026,
            'organismo_emisor': 'Ministerio de Salud Pública',
            'goc_code': 'GOC-2026-300-O40',
            'raw_text': 'RESOLUCION 41/2026 DE SALUD...',
        })
        conn = _make_in_memory_db(normas)
        index = build_norma_index(conn)
        ref = NormaReference(
            source_norma_id="other",
            ref_tipo="Resolución",
            ref_numero="41",
            ref_year=2026,
            ref_organismo=None,
        )
        resolved = resolve_references([ref], index)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].confidence, 'ambiguous')
        conn.close()

    def test_source_organismo_resolves_ambiguous(self):
        """When ambiguous, source norma's organismo can disambiguate."""
        normas = _sample_normas()
        normas.append({
            'norma_id': 'resolucion_41_2026_ministerio_de_salud_publica',
            'tipo': 'Resolución', 'numero': '41', 'year': 2026,
            'organismo_emisor': 'Ministerio de Salud Pública',
            'goc_code': 'GOC-2026-300-O40',
            'raw_text': 'RESOLUCION 41/2026 DE SALUD...',
        })
        conn = _make_in_memory_db(normas)
        index = build_norma_index(conn)
        ref = NormaReference(
            source_norma_id="resolucion_other_2026_ministerio_de_finanzas_y_precios",
            ref_tipo="Resolución",
            ref_numero="41",
            ref_year=2026,
            ref_organismo=None,
        )
        # Source norma is from MFP, so it should resolve to MFP's Res 41
        source_orgs = {
            "resolucion_other_2026_ministerio_de_finanzas_y_precios":
                "Ministerio de Finanzas y Precios",
        }
        resolved = resolve_references([ref], index, source_organismos=source_orgs)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].confidence, 'source_org')
        self.assertIn("finanzas", resolved[0].resolved_norma_id)
        conn.close()

    def test_source_organismo_still_ambiguous_if_multiple_match(self):
        """Source org heuristic doesn't help if 2+ candidates share the org."""
        normas = _sample_normas()
        # Add a second Res 41/2026 from the SAME organismo
        normas.append({
            'norma_id': 'resolucion_41_2026_ministerio_de_finanzas_y_precios_v2',
            'tipo': 'Resolución', 'numero': '41', 'year': 2026,
            'organismo_emisor': 'Ministerio de Finanzas y Precios',
            'goc_code': 'GOC-2026-999-O99',
            'raw_text': 'RESOLUCION 41/2026 v2...',
        })
        conn = _make_in_memory_db(normas)
        index = build_norma_index(conn)
        ref = NormaReference(
            source_norma_id="other",
            ref_tipo="Resolución",
            ref_numero="41",
            ref_year=2026,
            ref_organismo=None,
        )
        source_orgs = {"other": "Ministerio de Finanzas y Precios"}
        resolved = resolve_references([ref], index, source_organismos=source_orgs)
        self.assertEqual(len(resolved), 1)
        # Both candidates match the source org, so still ambiguous
        self.assertEqual(resolved[0].confidence, 'ambiguous')
        conn.close()


# ── Build Index ──────────────────────────────────────────────────────────────

class TestBuildNormaIndex(unittest.TestCase):

    def test_builds_lookup(self):
        conn = _make_in_memory_db(_sample_normas())
        index = build_norma_index(conn)
        self.assertIn('lookup', index)
        self.assertIn('loose_lookup', index)
        self.assertIn('organismos', index)
        self.assertIn('goc_codes', index)
        # Check a known entry
        key = ("resolucion", "41", 2026)
        self.assertIn(key, index['lookup'])
        # Loose lookup ignores year
        loose_key = ("resolucion", "41")
        self.assertIn(loose_key, index['loose_lookup'])
        conn.close()

    def test_empty_db(self):
        conn = _make_in_memory_db([])
        index = build_norma_index(conn)
        self.assertEqual(len(index['lookup']), 0)
        self.assertEqual(len(index['loose_lookup']), 0)
        conn.close()


# ── Loose Reference Extraction ───────────────────────────────────────────────

class TestExtractLooseReferences(unittest.TestCase):

    def test_dejar_sin_efecto_acuerdo(self):
        text = "SEGUNDA: Dejar sin efecto el Acuerdo 10085 del Consejo de Ministros."
        refs = extract_loose_references("source_id", text)
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].ref_numero, "10085")
        self.assertEqual(refs[0].relation_type, 'deroga')
        self.assertIsNone(refs[0].ref_year)

    def test_se_deroga_resolucion(self):
        text = "PRIMERA: Se deroga la Resolución 209 del Ministerio de la Construcción."
        refs = extract_loose_references("source_id", text)
        self.assertEqual(len(refs), 1)
        self.assertIn("esoluci", refs[0].ref_tipo)
        self.assertEqual(refs[0].ref_numero, "209")

    def test_se_modifica_ley(self):
        text = "TERCERA: Se modifica el artículo 5 de la Ley 16."
        refs = extract_loose_references("source_id", text)
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].relation_type, 'modifica')

    def test_no_action_verb_no_match(self):
        text = "Conforme a la Resolución 41 del Ministerio de Finanzas."
        refs = extract_loose_references("source_id", text)
        self.assertEqual(len(refs), 0)

    def test_skips_if_already_found_with_year(self):
        text = "PRIMERA: Se deroga la Resolución 41 del MFP."
        already = {("resolucion", "41")}
        refs = extract_loose_references("source_id", text, already_found=already)
        self.assertEqual(len(refs), 0)

    def test_self_reference_excluded(self):
        text = "PRIMERA: Se deroga la Resolución 41 del Ministerio."
        refs = extract_loose_references("resolucion_41_2026_mfp", text)
        self.assertEqual(len(refs), 0)

    def test_multiple_loose_refs(self):
        text = (
            "PRIMERA: Se derogan los Acuerdos 5204 y 5209 "
            "del Comité Ejecutivo del Consejo de Ministros."
        )
        refs = extract_loose_references("source_id", text)
        numeros = {r.ref_numero for r in refs}
        self.assertIn("5204", numeros)
        self.assertIn("5209", numeros)


class TestResolveLooseReferences(unittest.TestCase):

    def test_unique_match(self):
        """Loose ref with only one candidate in DB resolves."""
        conn = _make_in_memory_db(_sample_normas())
        index = build_norma_index(conn)
        ref = NormaReference(
            source_norma_id="other",
            ref_tipo="Ley",
            ref_numero="177",
            ref_year=None,
        )
        resolved = resolve_loose_references([ref], index)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].confidence, 'exact_loose')
        self.assertIn("ley_177", resolved[0].resolved_norma_id)
        conn.close()

    def test_ambiguous_loose(self):
        """Loose ref with multiple candidates stays ambiguous."""
        normas = _sample_normas()
        # Add a second Ley 177 from a different year
        normas.append({
            'norma_id': 'ley_177_2020_asamblea_nacional',
            'tipo': 'Ley', 'numero': '177', 'year': 2020,
            'organismo_emisor': 'Asamblea Nacional del Poder Popular',
            'goc_code': 'GOC-2020-50-O5',
            'raw_text': 'LEY 177 DE 2020...',
        })
        conn = _make_in_memory_db(normas)
        index = build_norma_index(conn)
        ref = NormaReference(
            source_norma_id="other",
            ref_tipo="Ley",
            ref_numero="177",
            ref_year=None,
        )
        resolved = resolve_loose_references([ref], index)
        self.assertEqual(len(resolved), 1)
        # Both candidates have same organismo, so source_org won't help
        # without specifying source_organismos
        self.assertEqual(resolved[0].confidence, 'ambiguous')
        conn.close()

    def test_source_org_disambiguates_loose(self):
        """Source organismo resolves ambiguous loose ref."""
        normas = _sample_normas()
        normas.append({
            'norma_id': 'resolucion_8_2025_ministerio_de_finanzas_y_precios',
            'tipo': 'Resolución', 'numero': '8', 'year': 2025,
            'organismo_emisor': 'Ministerio de Finanzas y Precios',
            'goc_code': 'GOC-2025-50-O5',
            'raw_text': 'RESOLUCION 8/2025...',
        })
        conn = _make_in_memory_db(normas)
        index = build_norma_index(conn)
        ref = NormaReference(
            source_norma_id="src_norma",
            ref_tipo="Resolución",
            ref_numero="8",
            ref_year=None,
        )
        # Source norma is from Ministerio de Economía → should resolve to Res 8/2026 from Economía
        source_orgs = {"src_norma": "Ministerio de Economía y Planificación"}
        resolved = resolve_loose_references([ref], index, source_organismos=source_orgs)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].confidence, 'source_org_loose')
        self.assertIn("economia", resolved[0].resolved_norma_id)
        conn.close()


# ── Integration ──────────────────────────────────────────────────────────────

class TestIntegration(unittest.TestCase):

    def test_end_to_end_small_dataset(self):
        """Process synthetic normas with known relationships."""
        normas = _sample_normas()
        # The first norma (Res 41) references Decreto-Ley 114
        normas[0]['raw_text'] = (
            "RESOLUCION 41/2026 POR CUANTO: El Ministerio...\n"
            "ARTICULO 1: Algo.\n"
            "DISPOSICIONES FINALES\n"
            "PRIMERA: Se modifica el Decreto-Ley 114 de 2025 del Consejo de Estado.\n"
            "SEGUNDA: La presente entra en vigor."
        )
        conn = _make_in_memory_db(normas)
        result = link_all_normas(conn)
        self.assertIsInstance(result, LinkingResult)
        self.assertGreaterEqual(result.total_references_found, 1)
        self.assertGreaterEqual(result.total_resolved, 1)
        # Check the relationship was created
        self.assertGreaterEqual(len(result.relationships), 1)
        rel = result.relationships[0]
        self.assertIn("decreto_ley_114", rel.target_norma_id)
        self.assertEqual(rel.relation_type, 'modifica')
        conn.close()

    def test_sqlite_tables_created(self):
        conn = _make_in_memory_db(_sample_normas())
        result = link_all_normas(conn)
        save_relationships_sqlite(result, conn)
        # Check tables exist
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {t[0] for t in tables}
        self.assertIn('norma_relationships', table_names)
        self.assertIn('norma_unresolved_references', table_names)
        conn.close()

    def test_sqlite_relationships_stored(self):
        normas = _sample_normas()
        normas[0]['raw_text'] = (
            "RESOLUCION 41/2026 POR CUANTO...\n"
            "DISPOSICIONES FINALES\n"
            "PRIMERA: Se deroga la Ley 177 de 2025 de la Asamblea Nacional."
        )
        conn = _make_in_memory_db(normas)
        result = link_all_normas(conn)
        save_relationships_sqlite(result, conn)
        count = conn.execute("SELECT COUNT(*) FROM norma_relationships").fetchone()[0]
        self.assertGreaterEqual(count, 1)
        conn.close()

    def test_report_structure(self):
        conn = _make_in_memory_db(_sample_normas())
        result = link_all_normas(conn)
        d = result.to_dict()
        self.assertIn('stats', d)
        self.assertIn('relationships', d)
        self.assertIn('unresolved_references', d)
        self.assertIn('total_references_found', d['stats'])
        conn.close()

    def test_empty_db_graceful(self):
        conn = _make_in_memory_db([])
        result = link_all_normas(conn)
        self.assertEqual(result.total_references_found, 0)
        self.assertEqual(len(result.relationships), 0)
        conn.close()


if __name__ == '__main__':
    unittest.main()
