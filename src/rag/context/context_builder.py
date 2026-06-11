from __future__ import annotations

import logging
from collections.abc import Sequence

from ..store.models import RetrievedFragment

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Builds the source context sent to the generator."""

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

    def build(self, fragments: Sequence[RetrievedFragment]) -> str:
        selected = self._select_fragments(fragments)
        if not selected:
            return ""

        context_parts: list[str] = []
        total_chars = 0

        for index, fragment in enumerate(selected, start=1):
            block = self._format_fragment(fragment, index)
            if self._reached_char_limit(total_chars, len(block), context_parts):
                logger.info(
                    "Limite de %d chars alcanzado en fuente %d",
                    self.max_chars,
                    index,
                )
                break

            context_parts.append(block)
            total_chars += len(block)

        logger.info(
            "Contexto construido: %d fuentes, %d chars",
            len(context_parts),
            total_chars,
        )
        return "\n".join(context_parts)

    def _select_fragments(
        self,
        fragments: Sequence[RetrievedFragment],
    ) -> list[RetrievedFragment]:
        selected = list(fragments[: self.max_fragments])
        if not self.deduplicate_by_norma:
            return selected

        seen: set[str] = set()
        deduplicated: list[RetrievedFragment] = []
        for fragment in selected:
            if fragment.norma_id in seen:
                continue
            seen.add(fragment.norma_id)
            deduplicated.append(fragment)

        return deduplicated

    def _format_fragment(self, fragment: RetrievedFragment, index: int) -> str:
        return f"{self._build_header(fragment, index)}\n---\n{fragment.content}\n"

    def _reached_char_limit(
        self,
        current_chars: int,
        next_block_chars: int,
        context_parts: list[str],
    ) -> bool:
        has_context = bool(context_parts)
        would_exceed_limit = current_chars + next_block_chars > self.max_chars
        return has_context and would_exceed_limit

    def _build_header(self, fragment: RetrievedFragment, index: int) -> str:
        header_lines = [
            (
                f"[Fuente {index}] {fragment.tipo} "
                f"{fragment.numero}/{fragment.year} - {fragment.organismo_emisor}"
            )
        ]

        if not self.include_metadata:
            return "\n".join(header_lines)

        metadata = self._metadata_parts(fragment)
        if metadata:
            header_lines.append(" | ".join(metadata))

        if fragment.fragment_label:
            header_lines.append(f"Seccion: {fragment.fragment_label}")

        return "\n".join(header_lines)

    def _metadata_parts(self, fragment: RetrievedFragment) -> list[str]:
        metadata = []
        if fragment.gaceta_numero:
            metadata.append(f"Gaceta Oficial No. {fragment.gaceta_numero}")
        if fragment.gaceta_fecha:
            metadata.append(fragment.gaceta_fecha)
        if fragment.goc_code:
            metadata.append(fragment.goc_code)

        page_range = self._format_page_range(fragment)
        if page_range:
            metadata.append(page_range)

        return metadata

    @staticmethod
    def _format_page_range(fragment: RetrievedFragment) -> str:
        page_start, page_end = fragment.page_range
        if page_start <= 0:
            return ""
        if page_start == page_end:
            return f"Pag. {page_start}"
        return f"Pags. {page_start}-{page_end}"
