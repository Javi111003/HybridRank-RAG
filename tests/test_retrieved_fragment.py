import pytest

from src.rag.store.models import RetrievedFragment


def _make_fragment(**overrides):
    defaults = {
        "fragment_id": "decreto_ley_114_2025_consejo_de_estado__abc123__GOC-2026-215-O24__f001",
        "content": "Texto de prueba del fragmento normativo.",
        "score": 0.85,
        "rank": 1,
        "metadata": {
            "norma_id": "decreto_ley_114_2025_consejo_de_estado",
            "tipo": "Decreto-Ley",
            "numero": "114",
            "year": 2025,
            "organismo_emisor": "Consejo de Estado",
            "goc_code": "GOC-2026-215-O24",
            "gaceta_numero": "24",
            "gaceta_fecha": "03 Marzo, 2026",
            "gaceta_tipo_edicion": "Ordinaria",
            "gaceta_pdf_url": "https://example.com/gaceta.pdf",
            "fragment_label": "Articulo 5",
            "match_confidence": "high",
            "page_start": 3,
            "page_end": 8,
        },
    }
    defaults.update(overrides)
    return RetrievedFragment(**defaults)


class TestRetrievedFragment:
    def test_properties_from_metadata(self):
        frag = _make_fragment()
        assert frag.norma_id == "decreto_ley_114_2025_consejo_de_estado"
        assert frag.tipo == "Decreto-Ley"
        assert frag.numero == "114"
        assert frag.year == 2025
        assert frag.organismo_emisor == "Consejo de Estado"
        assert frag.goc_code == "GOC-2026-215-O24"
        assert frag.gaceta_numero == "24"
        assert frag.gaceta_fecha == "03 Marzo, 2026"
        assert frag.fragment_label == "Articulo 5"
        assert frag.match_confidence == "high"
        assert frag.page_range == (3, 8)

    def test_citation_key(self):
        frag = _make_fragment()
        assert frag.citation_key() == "Decreto-Ley 114/2025 (GOC-2026-215-O24)"

    def test_citation_key_without_goc(self):
        frag = _make_fragment(metadata={"tipo": "Ley", "numero": "1", "year": 2020, "goc_code": ""})
        assert frag.citation_key() == "Ley 1/2020"

    def test_missing_metadata_fields(self):
        frag = _make_fragment(metadata={})
        assert frag.norma_id == ""
        assert frag.tipo == ""
        assert frag.year == 0
        assert frag.page_range == (-1, -1)
        assert frag.fragment_label == ""

    def test_year_type_coercion(self):
        frag = _make_fragment(metadata={"year": "2025"})
        assert frag.year == 2025

    def test_year_invalid_value(self):
        frag = _make_fragment(metadata={"year": "invalid"})
        assert frag.year == 0
