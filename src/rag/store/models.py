from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass
class RetrievedFragment:
    """Fragmento normativo recuperado con texto completo y metadata juridica."""

    fragment_id: str
    content: str
    score: float
    rank: int
    metadata: Dict[str, Any]

    @property
    def norma_id(self) -> str:
        return str(self.metadata.get("norma_id", ""))

    @property
    def tipo(self) -> str:
        return str(self.metadata.get("tipo", ""))

    @property
    def numero(self) -> str:
        return str(self.metadata.get("numero", ""))

    @property
    def year(self) -> int:
        try:
            return int(self.metadata.get("year", 0))
        except (ValueError, TypeError):
            return 0

    @property
    def organismo_emisor(self) -> str:
        return str(self.metadata.get("organismo_emisor", ""))

    @property
    def goc_code(self) -> str:
        return str(self.metadata.get("goc_code", ""))

    @property
    def gaceta_numero(self) -> str:
        return str(self.metadata.get("gaceta_numero", ""))

    @property
    def gaceta_fecha(self) -> str:
        return str(self.metadata.get("gaceta_fecha", ""))

    @property
    def gaceta_tipo_edicion(self) -> str:
        return str(self.metadata.get("gaceta_tipo_edicion", ""))

    @property
    def gaceta_pdf_url(self) -> str:
        return str(self.metadata.get("gaceta_pdf_url", ""))

    @property
    def fragment_label(self) -> str:
        return str(self.metadata.get("fragment_label", ""))

    @property
    def match_confidence(self) -> str:
        return str(self.metadata.get("match_confidence", ""))

    @property
    def page_range(self) -> Tuple[int, int]:
        try:
            start = int(self.metadata.get("page_start", -1))
        except (ValueError, TypeError):
            start = -1
        try:
            end = int(self.metadata.get("page_end", -1))
        except (ValueError, TypeError):
            end = -1
        return (start, end)

    def citation_key(self) -> str:
        parts = [f"{self.tipo} {self.numero}/{self.year}"]
        if self.goc_code:
            parts.append(f"({self.goc_code})")
        return " ".join(parts)
