from __future__ import annotations

import logging
import re
from typing import List, Set

from ..store.models import RetrievedFragment

logger = logging.getLogger(__name__)

_FUENTE_PATTERN = re.compile(r"\[Fuente\s+(\d+)\]", re.IGNORECASE)


class CitationFormatter:
    """Formatea fuentes juridicas y valida citas en las respuestas del LLM."""

    def format(
        self,
        answer_text: str,
        fragments: List[RetrievedFragment],
    ) -> str:
        cited_indices = self._extract_cited_sources(answer_text)

        valid_indices = set(range(1, len(fragments) + 1))
        invalid = cited_indices - valid_indices
        if invalid:
            logger.warning("El LLM cito fuentes inexistentes: %s", sorted(invalid))

        bibliography = self._build_bibliography(fragments, cited_indices & valid_indices)

        return f"{answer_text}\n\n{bibliography}"

    def format_source_label(self, frag: RetrievedFragment) -> str:
        parts = [f"{frag.tipo} {frag.numero}/{frag.year}"]
        if frag.organismo_emisor:
            parts.append(f"— {frag.organismo_emisor}")
        return " ".join(parts)

    def format_sources_list(self, fragments: List[RetrievedFragment]) -> str:
        lines = []
        for i, frag in enumerate(fragments, start=1):
            lines.append(f"[Fuente {i}] {self.format_source_label(frag)}")
        return "\n".join(lines)

    def _extract_cited_sources(self, text: str) -> Set[int]:
        return {int(m) for m in _FUENTE_PATTERN.findall(text)}

    def _build_bibliography(
        self,
        fragments: List[RetrievedFragment],
        cited_indices: Set[int],
    ) -> str:
        lines = ["---", "**Fuentes Consultadas**", ""]

        for i, frag in enumerate(fragments, start=1):
            if i in cited_indices:
                lines.append(self._format_source_entry(frag, i))
                lines.append("")

        uncited = [
            i for i in range(1, len(fragments) + 1) if i not in cited_indices
        ]
        if uncited:
            lines.append("**Otras fuentes recuperadas:**")
            for i in uncited:
                frag = fragments[i - 1]
                lines.append(f"- {frag.citation_key()}")
            lines.append("")

        return "\n".join(lines)

    def _format_source_entry(self, frag: RetrievedFragment, index: int) -> str:
        parts = [
            f"**[Fuente {index}]** {frag.tipo} {frag.numero}/{frag.year}",
            f"  *{frag.organismo_emisor}*",
        ]

        meta_parts: list[str] = []
        if frag.gaceta_numero and frag.gaceta_fecha:
            meta_parts.append(
                f"Gaceta Oficial No. {frag.gaceta_numero}, {frag.gaceta_fecha}"
            )
        if frag.goc_code:
            meta_parts.append(frag.goc_code)

        page_start, page_end = frag.page_range
        if page_start > 0:
            if page_start == page_end:
                meta_parts.append(f"Pag. {page_start}")
            else:
                meta_parts.append(f"Pags. {page_start}-{page_end}")

        if meta_parts:
            parts.append(f"  {' | '.join(meta_parts)}")

        return "\n".join(parts)
