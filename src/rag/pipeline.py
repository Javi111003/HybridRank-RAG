from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.config import config
from src.retriever import Retriever
from .store.norma_store import NormaStore
from .store.models import RetrievedFragment
from .context.context_builder import ContextBuilder
from .prompt.prompt_builder import PromptBuilder
from .generator.base import GeneratorProvider, GenerationResult
from .generator.registry import get_generator
from .citation.citation_formatter import CitationFormatter

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """Resultado completo del pipeline RAG."""

    query: str
    answer: str
    raw_answer: str
    fragments: List[RetrievedFragment]
    generation_result: GenerationResult
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    retriever_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class RAGPipeline:
    """Pipeline completo: retrieval -> context -> prompt -> generation -> citation."""

    def __init__(
        self,
        retriever: Retriever,
        store: NormaStore | None = None,
        context_builder: ContextBuilder | None = None,
        prompt_builder: PromptBuilder | None = None,
        generator: GeneratorProvider | None = None,
        citation_formatter: CitationFormatter | None = None,
        top_k: int | None = None,
        log_interactions: bool = True,
    ):
        self._retriever = retriever
        self._store = store or NormaStore()
        self._context_builder = context_builder or ContextBuilder(
            max_fragments=config.CONTEXT_MAX_FRAGMENTS,
            max_chars=config.CONTEXT_MAX_CHARS,
        )
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._generator = generator or get_generator()
        self._citation_formatter = citation_formatter or CitationFormatter()
        self._top_k = top_k or config.TOP_K
        self._log_interactions = log_interactions

        logger.info(
            "RAGPipeline inicializado: retriever=%s, generator=%s, top_k=%d",
            retriever.name,
            self._generator.name,
            self._top_k,
        )

    def run(self, query: str, top_k: int | None = None) -> RAGResult:
        k = top_k or self._top_k
        pipeline_start = time.time()

        logger.info("RAG pipeline: query='%s', top_k=%d", query, k)

        # 1. Retrieval
        t0 = time.time()
        retrieval_results = self._retriever.retrieve(query, top_k=k)
        retrieval_ms = (time.time() - t0) * 1000
        logger.info(
            "Retrieval: %d resultados en %.0fms", len(retrieval_results), retrieval_ms
        )

        if not retrieval_results:
            logger.warning("Sin resultados de retrieval")
            result = self._empty_result(query, retrieval_ms)
            self._log_interaction(result)
            return result

        # 2. Resolver fragmentos via NormaStore
        fragments = self._store.get_fragments(retrieval_results)
        if not fragments:
            logger.warning("Sin fragmentos resueltos desde NormaStore")
            result = self._empty_result(query, retrieval_ms)
            self._log_interaction(result)
            return result

        # 3. Construir contexto
        context = self._context_builder.build(fragments)
        logger.info("Contexto: %d chars, %d fragmentos", len(context), len(fragments))

        # 4. Construir prompt
        messages = self._prompt_builder.build(query, context)

        # 5. Generar respuesta
        t1 = time.time()
        generation_result = self._generator.generate(messages)
        generation_ms = (time.time() - t1) * 1000
        logger.info(
            "Generacion: %.0fms, %d chars",
            generation_ms,
            len(generation_result.text),
        )

        # 6. Formatear citas
        answer = self._citation_formatter.format(generation_result.text, fragments)

        total_ms = (time.time() - pipeline_start) * 1000
        logger.info("Pipeline completo: %.0fms total", total_ms)

        result = RAGResult(
            query=query,
            answer=answer,
            raw_answer=generation_result.text,
            fragments=fragments,
            generation_result=generation_result,
            retrieval_time_ms=retrieval_ms,
            generation_time_ms=generation_ms,
            total_time_ms=total_ms,
            retriever_name=self._retriever.name,
            metadata={
                "top_k": k,
                "num_fragments": len(fragments),
                "context_chars": len(context),
            },
        )

        self._log_interaction(result)
        return result

    def _empty_result(self, query: str, retrieval_ms: float) -> RAGResult:
        empty_gen = GenerationResult(
            text="No se encontraron fuentes relevantes para responder la consulta.",
            model="none",
        )
        return RAGResult(
            query=query,
            answer=empty_gen.text,
            raw_answer=empty_gen.text,
            fragments=[],
            generation_result=empty_gen,
            retrieval_time_ms=retrieval_ms,
            generation_time_ms=0.0,
            total_time_ms=retrieval_ms,
            retriever_name=self._retriever.name,
        )

    def _log_interaction(self, result: RAGResult) -> None:
        if not self._log_interactions:
            return

        try:
            log_dir = config.LOG_DIR
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "rag_interactions.jsonl")

            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "query": result.query,
                "retriever": result.retriever_name,
                "provider": result.generation_result.model,
                "retrieved_docs": [
                    {"fragment_id": f.fragment_id, "score": f.score}
                    for f in result.fragments
                ],
                "sources": [f.citation_key() for f in result.fragments],
                "answer": result.raw_answer[:500],
                "usage": result.generation_result.usage,
                "retrieval_time_ms": result.retrieval_time_ms,
                "generation_time_ms": result.generation_time_ms,
                "total_time_ms": result.total_time_ms,
            }

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("Error guardando log de interaccion: %s", e)
