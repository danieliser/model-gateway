"""In-process cross-encoder reranking using sentence-transformers.

Manages model loading/unloading and provides a rerank() function
that returns Cohere-compatible rerank responses. No subprocess,
no extra port — runs inside the gateway process.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class _LoadedRerankModel:
    model: Any  # CrossEncoder instance
    model_name: str
    last_used: float = field(default_factory=time.monotonic)


class RerankManager:
    """Manages in-process cross-encoder reranking models."""

    def __init__(self) -> None:
        self._models: dict[str, _LoadedRerankModel] = {}

    def is_loaded(self, model_alias: str) -> bool:
        return model_alias in self._models

    def load(self, model_alias: str, model_path: str) -> None:
        """Load a reranking model into memory."""
        if model_alias in self._models:
            self._models[model_alias].last_used = time.monotonic()
            return

        logger.info("Loading reranker model '%s' from %s", model_alias, model_path)

        from sentence_transformers import CrossEncoder

        model = CrossEncoder(model_path, trust_remote_code=True)

        self._models[model_alias] = _LoadedRerankModel(
            model=model,
            model_name=model_path.rstrip("/").rsplit("/", 1)[-1],
        )
        logger.info("Reranker model '%s' loaded", model_alias)

    def unload(self, model_alias: str) -> bool:
        """Unload a reranking model from memory. Returns True if unloaded."""
        info = self._models.pop(model_alias, None)
        if info is None:
            return False
        del info.model
        logger.info("Reranker model '%s' unloaded", model_alias)
        return True

    def touch(self, model_alias: str) -> None:
        """Update last_used timestamp."""
        info = self._models.get(model_alias)
        if info:
            info.last_used = time.monotonic()

    def get_last_used(self, model_alias: str) -> float | None:
        info = self._models.get(model_alias)
        return info.last_used if info else None

    def rerank(
        self,
        model_alias: str,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> dict:
        """Rerank documents by relevance to query.

        Returns a Cohere-compatible response dict with results sorted by
        descending relevance_score.
        """
        info = self._models.get(model_alias)
        if info is None:
            raise RuntimeError(f"Reranker model '{model_alias}' not loaded")

        info.last_used = time.monotonic()

        if not documents:
            return {"results": [], "model": info.model_name}

        # CrossEncoder.predict expects list of [query, document] pairs
        pairs = [[query, doc] for doc in documents]
        scores = info.model.predict(pairs)

        # Build indexed results, sort by score descending
        indexed = [
            {"index": i, "relevance_score": float(s)}
            for i, s in enumerate(scores)
        ]
        indexed.sort(key=lambda x: x["relevance_score"], reverse=True)

        if top_n is not None:
            indexed = indexed[:top_n]

        return {
            "results": indexed,
            "model": info.model_name,
        }
