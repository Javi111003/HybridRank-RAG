import pytest
from typing import List, Tuple
from unittest.mock import MagicMock, patch

from src.retriever.retriever import Retriever
from src.rag.store.models import RetrievedFragment
from src.rag.generator.base import GeneratorProvider, GenerationResult
from src.rag.context.context_builder import ContextBuilder
from src.rag.prompt.prompt_builder import PromptBuilder
from src.rag.citation.citation_formatter import CitationFormatter
from src.rag.pipeline import RAGPipeline, RAGResult


class FakeRetriever(Retriever):
    def __init__(self, results=None):
        self._results = results or []

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        return self._results[:top_k]

    @property
    def name(self) -> str:
        return "FakeRetriever"


class FakeGenerator(GeneratorProvider):
    def __init__(self, text="Segun [Fuente 1], el decreto establece regulaciones."):
        self._text = text

    def generate(self, messages):
        return GenerationResult(
            text=self._text,
            model="fake-model",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            finish_reason="stop",
        )

    @property
    def name(self) -> str:
        return "FakeGenerator"


def _make_fragment(idx):
    return RetrievedFragment(
        fragment_id=f"frag_{idx:03d}",
        content=f"Contenido normativo del fragmento {idx}.",
        score=1.0 - idx * 0.1,
        rank=idx,
        metadata={
            "norma_id": f"norma_{idx}",
            "tipo": "Decreto-Ley",
            "numero": str(100 + idx),
            "year": 2025,
            "organismo_emisor": "Consejo de Estado",
            "goc_code": f"GOC-2026-{idx:03d}-O01",
            "gaceta_numero": str(idx),
            "gaceta_fecha": f"0{idx} Enero, 2026",
            "page_start": idx,
            "page_end": idx + 5,
        },
    )


class FakeNormaStore:
    def __init__(self, fragments):
        self._fragments = {f.fragment_id: f for f in fragments}

    def get_fragments(self, retrieval_results):
        result = []
        rank = 1
        for fid, score in retrieval_results:
            if fid in self._fragments:
                frag = self._fragments[fid]
                result.append(RetrievedFragment(
                    fragment_id=frag.fragment_id,
                    content=frag.content,
                    score=score,
                    rank=rank,
                    metadata=frag.metadata,
                ))
                rank += 1
        return result


class TestRAGPipeline:
    def _build_pipeline(self, retriever_results, generator_text=None):
        fragments = [_make_fragment(i) for i in range(1, 4)]
        retriever = FakeRetriever(retriever_results)
        store = FakeNormaStore(fragments)
        generator = FakeGenerator(
            text=generator_text or "Segun [Fuente 1], el decreto establece regulaciones."
        )

        pipeline = RAGPipeline(
            retriever=retriever,
            store=store,
            context_builder=ContextBuilder(max_fragments=5),
            prompt_builder=PromptBuilder(),
            generator=generator,
            citation_formatter=CitationFormatter(),
            top_k=10,
            log_interactions=False,
        )
        return pipeline

    def test_full_pipeline(self):
        results = [("frag_001", 0.9), ("frag_002", 0.7), ("frag_003", 0.5)]
        pipeline = self._build_pipeline(results)

        rag_result = pipeline.run("que dice el decreto 101")

        assert isinstance(rag_result, RAGResult)
        assert rag_result.query == "que dice el decreto 101"
        assert "Segun [Fuente 1]" in rag_result.raw_answer
        assert "Fuentes Consultadas" in rag_result.answer
        assert len(rag_result.fragments) == 3
        assert rag_result.retrieval_time_ms >= 0
        assert rag_result.generation_time_ms >= 0
        assert rag_result.total_time_ms >= 0
        assert rag_result.retriever_name == "FakeRetriever"

    def test_empty_retrieval(self):
        pipeline = self._build_pipeline([])
        result = pipeline.run("consulta sin resultados")

        assert "No se encontraron fuentes" in result.answer
        assert result.fragments == []
        assert result.generation_result.model == "none"

    def test_partial_store_resolution(self):
        results = [("frag_001", 0.9), ("nonexistent", 0.5)]
        pipeline = self._build_pipeline(results)
        result = pipeline.run("consulta parcial")

        assert len(result.fragments) == 1
        assert result.fragments[0].fragment_id == "frag_001"

    def test_pipeline_preserves_scores(self):
        results = [("frag_001", 0.95), ("frag_002", 0.72)]
        pipeline = self._build_pipeline(results)
        result = pipeline.run("consulta con scores")

        assert result.fragments[0].score == 0.95
        assert result.fragments[1].score == 0.72

    def test_pipeline_custom_top_k(self):
        results = [("frag_001", 0.9), ("frag_002", 0.7), ("frag_003", 0.5)]
        pipeline = self._build_pipeline(results)
        result = pipeline.run("consulta", top_k=1)

        assert len(result.fragments) == 1

    def test_generation_result_metadata(self):
        results = [("frag_001", 0.9)]
        pipeline = self._build_pipeline(results)
        result = pipeline.run("consulta")

        assert result.generation_result.model == "fake-model"
        assert result.generation_result.usage["total_tokens"] == 150
        assert result.generation_result.finish_reason == "stop"

    def test_metadata_in_result(self):
        results = [("frag_001", 0.9), ("frag_002", 0.7)]
        pipeline = self._build_pipeline(results)
        result = pipeline.run("consulta")

        assert result.metadata["num_fragments"] == 2
        assert result.metadata["top_k"] == 10
        assert result.metadata["context_chars"] > 0
