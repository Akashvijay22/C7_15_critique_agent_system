"""
Design knowledge base — the RAG layer, backed by LanceDB.

Design pattern: **Repository**. The rest of the app talks to ``KnowledgeBase``
and never touches LanceDB or the embedding model directly. That keeps the vector
store swappable and the agents testable.

What it does:
- On first run, embeds the seed corpus (Nielsen, UX laws, WCAG, CRO) into a
  LanceDB table using a local sentence-transformers model (no API key needed).
- ``retrieve(query, category)`` returns the most relevant principles for an
  agent to cite.
- ``embed(texts)`` exposes the same embedder so the coordinator can measure
  semantic similarity between findings for de-duplication.
"""

from __future__ import annotations

import logging
from functools import cached_property

import lancedb
import numpy as np
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

from config import Settings
from src.core.schemas import Citation
from src.rag.seed_data import SEED_KNOWLEDGE

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """Repository over a LanceDB table of grounded design principles."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._db = lancedb.connect(settings.lancedb_path)
        # Local embedding model from LanceDB's registry: runs in-process, no API.
        self._embedder = (
            get_registry()
            .get("sentence-transformers")
            .create(name=settings.embedding_model)
        )
        self._table = self._ensure_table()

    @cached_property
    def _schema(self) -> type[LanceModel]:
        embedder = self._embedder

        class DesignPrinciple(LanceModel):
            # `text` is the source field; LanceDB auto-embeds it into `vector`.
            text: str = embedder.SourceField()
            vector: Vector(embedder.ndims()) = embedder.VectorField()
            kb_id: str
            category: str
            title: str
            source: str
            url: str

        return DesignPrinciple

    def _ensure_table(self):
        name = self._settings.kb_table
        if name in self._db.table_names():
            tbl = self._db.open_table(name)
            if tbl.count_rows() >= len(SEED_KNOWLEDGE):
                logger.info("Knowledge base already populated (%s rows).", tbl.count_rows())
                return tbl
            # stale/partial — rebuild cleanly
            self._db.drop_table(name)

        logger.info("Building knowledge base with %s principles...", len(SEED_KNOWLEDGE))
        tbl = self._db.create_table(name, schema=self._schema, mode="overwrite")
        tbl.add([dict(entry) for entry in SEED_KNOWLEDGE])
        return tbl

    # ----------------------------------------------------------------- retrieval
    def retrieve(self, query: str, category: str | None = None, k: int | None = None) -> list[Citation]:
        """Return the top-k principles most relevant to ``query``.

        If ``category`` is given, results are biased to that lens (visual / ux /
        accessibility / market) but the search remains semantic.
        """
        k = k or self._settings.retrieval_k
        search = self._table.search(query)
        if category:
            # over-fetch then filter so we still return k after the category cut
            rows = search.limit(k * 3).to_list()
            rows = [r for r in rows if r["category"] == category] or rows
            rows = rows[:k]
        else:
            rows = search.limit(k).to_list()

        return [
            Citation(kb_id=r["kb_id"], title=r["title"], source=r["source"], url=r.get("url"))
            for r in rows
        ]

    def context_block(self, query: str, category: str | None = None, k: int | None = None) -> tuple[str, list[Citation]]:
        """Build a prompt-ready context string plus the citations it came from."""
        k = k or self._settings.retrieval_k
        search = self._table.search(query)
        rows = search.limit((k * 3) if category else k).to_list()
        if category:
            filtered = [r for r in rows if r["category"] == category]
            rows = (filtered or rows)[:k]

        citations = [
            Citation(kb_id=r["kb_id"], title=r["title"], source=r["source"], url=r.get("url"))
            for r in rows
        ]
        block = "\n".join(
            f"[{r['kb_id']}] {r['title']}: {r['text']} (Source: {r['source']})"
            for r in rows
        )
        return block, citations

    # ------------------------------------------------------------- embeddings api
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed arbitrary text with the same model used for the KB.

        Used by the coordinator to cluster near-duplicate findings.
        """
        if not texts:
            return np.zeros((0, self._embedder.ndims()), dtype=np.float32)
        vectors = self._embedder.compute_source_embeddings(texts)
        return np.asarray(vectors, dtype=np.float32)
