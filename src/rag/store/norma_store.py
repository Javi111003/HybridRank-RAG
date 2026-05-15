from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import chromadb

from src.config import config
from .models import RetrievedFragment

logger = logging.getLogger(__name__)


class NormaStore:
    """Resuelve fragment_ids a contenido completo y metadata desde ChromaDB."""

    def __init__(
        self,
        chroma_dir: str | None = None,
        collection_name: str | None = None,
    ):
        self._chroma_dir = chroma_dir or config.CHROMA_NORMA_DIR
        self._collection_name = collection_name or config.CHROMA_NORMA_COLLECTION

        self._client = chromadb.PersistentClient(path=self._chroma_dir)
        self._collection = self._client.get_collection(name=self._collection_name)
        logger.info(
            "NormaStore inicializado: coleccion '%s' (%d documentos)",
            self._collection_name,
            self._collection.count(),
        )

    def get_fragments(
        self,
        retrieval_results: List[Tuple[str, float]],
    ) -> List[RetrievedFragment]:
        if not retrieval_results:
            return []

        ids = [fid for fid, _ in retrieval_results]
        scores = {fid: score for fid, score in retrieval_results}

        result = self._collection.get(
            ids=ids,
            include=["documents", "metadatas"],
        )

        returned_ids = set(result["ids"])
        missing = [fid for fid in ids if fid not in returned_ids]
        if missing:
            logger.warning("Fragment IDs no encontrados en ChromaDB: %s", missing)

        id_to_idx = {fid: i for i, fid in enumerate(result["ids"])}

        fragments = []
        rank = 1
        for fid, _ in retrieval_results:
            if fid not in id_to_idx:
                continue
            idx = id_to_idx[fid]
            fragments.append(
                RetrievedFragment(
                    fragment_id=fid,
                    content=result["documents"][idx],
                    score=scores[fid],
                    rank=rank,
                    metadata=result["metadatas"][idx] or {},
                )
            )
            rank += 1

        logger.info("Resueltos %d/%d fragmentos", len(fragments), len(ids))
        return fragments

    def get_fragment(self, fragment_id: str) -> Optional[RetrievedFragment]:
        result = self._collection.get(
            ids=[fragment_id],
            include=["documents", "metadatas"],
        )
        if not result["ids"]:
            return None
        return RetrievedFragment(
            fragment_id=fragment_id,
            content=result["documents"][0],
            score=0.0,
            rank=1,
            metadata=result["metadatas"][0] or {},
        )
