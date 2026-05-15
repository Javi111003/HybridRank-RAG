import pytest

from src.rag.store.models import RetrievedFragment
from src.rag.citation.citation_formatter import CitationFormatter


def _make_fragment(idx, **meta_overrides):
    metadata = {
        "norma_id": f"norma_{idx}",
        "tipo": "Resolucion",
        "numero": str(idx),
        "year": 2025,
        "organismo_emisor": "Ministerio de Pruebas",
        "goc_code": f"GOC-2026-{idx:03d}-O01",
        "gaceta_numero": "1",
        "gaceta_fecha": "01 Enero, 2026",
        "page_start": idx,
        "page_end": idx + 2,
    }
    metadata.update(meta_overrides)
    return RetrievedFragment(
        fragment_id=f"frag_{idx:03d}",
        content=f"Contenido del fragmento {idx}.",
        score=0.9 - idx * 0.1,
        rank=idx,
        metadata=metadata,
    )


class TestCitationFormatter:
    def test_format_with_cited_sources(self):
        formatter = CitationFormatter()
        fragments = [_make_fragment(1), _make_fragment(2)]
        answer = "Segun [Fuente 1], esto aplica. Ver tambien [Fuente 2]."

        result = formatter.format(answer, fragments)

        assert "Fuentes Consultadas" in result
        assert "[Fuente 1]" in result
        assert "[Fuente 2]" in result
        assert "Resolucion 1/2025" in result

    def test_format_with_uncited_sources(self):
        formatter = CitationFormatter()
        fragments = [_make_fragment(1), _make_fragment(2), _make_fragment(3)]
        answer = "Segun [Fuente 1], esto aplica."

        result = formatter.format(answer, fragments)

        assert "Otras fuentes recuperadas" in result

    def test_format_with_no_citations(self):
        formatter = CitationFormatter()
        fragments = [_make_fragment(1)]
        answer = "No hay citas en esta respuesta."

        result = formatter.format(answer, fragments)

        assert "Otras fuentes recuperadas" in result

    def test_format_invalid_citation_index(self):
        formatter = CitationFormatter()
        fragments = [_make_fragment(1)]
        answer = "Segun [Fuente 1] y [Fuente 99]."

        result = formatter.format(answer, fragments)
        assert "[Fuente 1]" in result

    def test_format_source_label(self):
        formatter = CitationFormatter()
        frag = _make_fragment(1)
        label = formatter.format_source_label(frag)
        assert "Resolucion 1/2025" in label
        assert "Ministerio de Pruebas" in label

    def test_format_sources_list(self):
        formatter = CitationFormatter()
        fragments = [_make_fragment(1), _make_fragment(2)]
        result = formatter.format_sources_list(fragments)
        assert "[Fuente 1]" in result
        assert "[Fuente 2]" in result

    def test_format_with_missing_metadata(self):
        formatter = CitationFormatter()
        frag = RetrievedFragment(
            fragment_id="frag_sparse",
            content="Texto sin metadata.",
            score=0.5,
            rank=1,
            metadata={"tipo": "Ley", "numero": "1", "year": 2020},
        )
        answer = "Segun [Fuente 1], la ley dice..."
        result = formatter.format(answer, [frag])

        assert "Ley 1/2020" in result

    def test_gaceta_info_in_entry(self):
        formatter = CitationFormatter()
        fragments = [_make_fragment(1)]
        answer = "[Fuente 1] aplica."
        result = formatter.format(answer, fragments)

        assert "Gaceta Oficial No. 1" in result
        assert "01 Enero, 2026" in result
        assert "GOC-2026-001-O01" in result
