from __future__ import annotations

import logging
from typing import List

from ..store.models import RetrievedFragment

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Construye contexto estructurado para el LLM desde fragmentos recuperados."""

    def __init__(
        self,
        max_fragments: int = 8,
        max_chars: int = 12000,
        include_metadata: bool = True,
        deduplicate_by_norma: bool = False,
    ):
        self.max_fragments = max_fragments
        self.max_chars = max_chars
        self.include_metadata = include_metadata
        self.deduplicate_by_norma = deduplicate_by_norma

    def build(self, fragments: List[RetrievedFragment]) -> str:
        if not fragments:
            return ""

        working = list(fragments[: self.max_fragments])

        if self.deduplicate_by_norma:
            seen: set[str] = set()
            deduplicated: list[RetrievedFragment] = []
            for frag in working:
                if frag.norma_id not in seen:
                    deduplicated.append(frag)
                    seen.add(frag.norma_id)
            working = deduplicated

        context_parts: list[str] = []
        total_chars = 0

        for i, frag in enumerate(working, start=1):
            header = self._build_header(frag, i)
            block = f"{header}\n---\n{frag.content}\n"
            block_len = len(block)

            if total_chars + block_len > self.max_chars and context_parts:
                logger.info(
                    "Limite de %d chars alcanzado en fuente %d", self.max_chars, i
                )
                break

            context_parts.append(block)
            total_chars += block_len

        result = "\n".join(context_parts)
        logger.info(
            "Contexto construido: %d fuentes, %d chars", len(context_parts), total_chars
        )
        return result

    def _build_header(self, frag: RetrievedFragment, index: int) -> str:
        header_lines = [
            f"[Fuente {index}] {frag.tipo} {frag.numero}/{frag.year} — {frag.organismo_emisor}"
        ]

        if self.include_metadata:
            meta_parts: list[str] = []
            if frag.gaceta_numero:
                meta_parts.append(f"Gaceta Oficial No. {frag.gaceta_numero}")
            if frag.gaceta_fecha:
                meta_parts.append(frag.gaceta_fecha)
            if frag.goc_code:
                meta_parts.append(frag.goc_code)

            page_start, page_end = frag.page_range
            if page_start > 0:
                if page_start == page_end:
                    meta_parts.append(f"Pag. {page_start}")
                else:
                    meta_parts.append(f"Pags. {page_start}-{page_end}")

            if meta_parts:
                header_lines.append(" | ".join(meta_parts))

            if frag.fragment_label:
                header_lines.append(f"Seccion: {frag.fragment_label}")

        return "\n".join(header_lines)
