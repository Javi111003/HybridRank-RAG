import pytest

from src.rag.store.models import RetrievedFragment
from src.rag.context.context_builder import ContextBuilder


def _make_fragment(idx, norma_id="norma_a", content="Contenido de prueba.", **meta_overrides):
    metadata = {
        "norma_id": norma_id,
        "tipo": "Resolucion",
        "numero": str(idx),
        "year": 2025,
        "organismo_emisor": "Ministerio de Pruebas",
        "goc_code": f"GOC-2026-{idx:03d}-O01",
        "gaceta_numero": "1",
        "gaceta_fecha": "01 Enero, 2026",
        "fragment_label": f"Articulo {idx}",
        "page_start": idx,
        "page_end": idx + 2,
    }
    metadata.update(meta_overrides)
    return RetrievedFragment(
        fragment_id=f"frag_{idx:03d}",
        content=content,
        score=1.0 - idx * 0.1,
        rank=idx,
        metadata=metadata,
    )


class TestContextBuilder:
    def test_build_basic(self):
        builder = ContextBuilder(max_fragments=5)
        fragments = [_make_fragment(1), _make_fragment(2)]
        result = builder.build(fragments)

        assert "[Fuente 1]" in result
        assert "[Fuente 2]" in result
        assert "Contenido de prueba." in result

    def test_build_empty_fragments(self):
        builder = ContextBuilder()
        assert builder.build([]) == ""

    def test_max_fragments_limit(self):
        builder = ContextBuilder(max_fragments=2)
        fragments = [_make_fragment(i) for i in range(1, 6)]
        result = builder.build(fragments)

        assert "[Fuente 1]" in result
        assert "[Fuente 2]" in result
        assert "[Fuente 3]" not in result

    def test_max_chars_limit(self):
        long_content = "A" * 5000
        builder = ContextBuilder(max_fragments=10, max_chars=6000)
        fragments = [
            _make_fragment(1, content=long_content),
            _make_fragment(2, content=long_content),
            _make_fragment(3, content=long_content),
        ]
        result = builder.build(fragments)

        assert "[Fuente 1]" in result
        assert "[Fuente 3]" not in result

    def test_metadata_in_header(self):
        builder = ContextBuilder(include_metadata=True)
        fragments = [_make_fragment(1)]
        result = builder.build(fragments)

        assert "Gaceta Oficial No. 1" in result
        assert "GOC-2026-001-O01" in result
        assert "Articulo 1" in result

    def test_no_metadata_in_header(self):
        builder = ContextBuilder(include_metadata=False)
        fragments = [_make_fragment(1)]
        result = builder.build(fragments)

        assert "Gaceta Oficial" not in result

    def test_deduplicate_by_norma(self):
        builder = ContextBuilder(deduplicate_by_norma=True, max_fragments=10)
        fragments = [
            _make_fragment(1, norma_id="norma_a"),
            _make_fragment(2, norma_id="norma_a"),
            _make_fragment(3, norma_id="norma_b"),
        ]
        result = builder.build(fragments)

        assert "[Fuente 1]" in result
        assert "[Fuente 2]" in result
        assert "[Fuente 3]" not in result

    def test_no_deduplication_by_default(self):
        builder = ContextBuilder(max_fragments=10)
        fragments = [
            _make_fragment(1, norma_id="norma_a"),
            _make_fragment(2, norma_id="norma_a"),
            _make_fragment(3, norma_id="norma_b"),
        ]
        result = builder.build(fragments)

        assert "[Fuente 1]" in result
        assert "[Fuente 2]" in result
        assert "[Fuente 3]" in result

    def test_page_range_display(self):
        builder = ContextBuilder()
        fragments = [_make_fragment(1, page_start=5, page_end=5)]
        result = builder.build(fragments)
        assert "Pag. 5" in result

        fragments = [_make_fragment(1, page_start=5, page_end=10)]
        result = builder.build(fragments)
        assert "Pags. 5-10" in result
