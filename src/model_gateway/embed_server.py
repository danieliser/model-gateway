"""In-process MLX embedding inference using mlx-embeddings.

Manages model loading/unloading and provides a generate() function
that returns OpenAI-compatible embedding responses. No subprocess,
no extra port — runs inside the gateway process.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Lazy import — resolved on first use
mx = None


def _ensure_mlx():
    global mx
    if mx is None:
        import mlx.core
        mx = mlx.core

# ---------------------------------------------------------------------------
# Loaded model state
# ---------------------------------------------------------------------------

@dataclass
class _LoadedEmbedModel:
    model: Any
    tokenizer: Any
    model_name: str
    last_used: float = field(default_factory=time.monotonic)


class EmbedManager:
    """Manages in-process MLX embedding models."""

    def __init__(self) -> None:
        self._models: dict[str, _LoadedEmbedModel] = {}

    def is_loaded(self, model_alias: str) -> bool:
        return model_alias in self._models

    def load(self, model_alias: str, model_path: str) -> None:
        """Load an embedding model into memory."""
        if model_alias in self._models:
            self._models[model_alias].last_used = time.monotonic()
            return

        resolved = os.path.expanduser(model_path)
        logger.info("Loading embedding model '%s' from %s", model_alias, resolved)

        from mlx_embeddings.utils import load
        model, tokenizer_wrapper = load(resolved)

        # Unwrap TokenizerWrapper to get the underlying callable tokenizer
        if hasattr(tokenizer_wrapper, "_tokenizer"):
            tokenizer = tokenizer_wrapper._tokenizer
        else:
            tokenizer = tokenizer_wrapper

        self._models[model_alias] = _LoadedEmbedModel(
            model=model,
            tokenizer=tokenizer,
            model_name=os.path.basename(resolved.rstrip("/")),
        )
        logger.info("Embedding model '%s' loaded", model_alias)

    def unload(self, model_alias: str) -> bool:
        """Unload an embedding model from memory. Returns True if unloaded."""
        info = self._models.pop(model_alias, None)
        if info is None:
            return False
        # Let Python GC reclaim the model memory
        del info.model
        del info.tokenizer
        logger.info("Embedding model '%s' unloaded", model_alias)
        return True

    def touch(self, model_alias: str) -> None:
        """Update last_used timestamp."""
        info = self._models.get(model_alias)
        if info:
            info.last_used = time.monotonic()

    def get_last_used(self, model_alias: str) -> float | None:
        info = self._models.get(model_alias)
        return info.last_used if info else None

    def generate(self, model_alias: str, input_text: str | list[str]) -> dict:
        """Generate embeddings. Returns an OpenAI-compatible response dict."""
        info = self._models.get(model_alias)
        if info is None:
            raise RuntimeError(f"Embedding model '{model_alias}' not loaded")

        info.last_used = time.monotonic()
        _ensure_mlx()
        texts = [input_text] if isinstance(input_text, str) else input_text

        encoded = info.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="np"
        )
        input_ids = mx.array(encoded["input_ids"])
        attention_mask = mx.array(encoded["attention_mask"])

        output = info.model(input_ids, attention_mask=attention_mask)

        # Prefer text_embeds (already pooled + normalized) if available
        if hasattr(output, "text_embeds") and output.text_embeds is not None:
            emb = output.text_embeds
        else:
            hs = output.last_hidden_state
            if hs.ndim == 3:
                mask_exp = mx.expand_dims(attention_mask, -1).astype(hs.dtype)
                emb = (hs * mask_exp).sum(axis=1) / mask_exp.sum(axis=1)
            else:
                emb = hs
            norms = mx.sqrt((emb * emb).sum(axis=-1, keepdims=True))
            emb = emb / mx.maximum(norms, mx.array(1e-12))

        mx.async_eval(emb)
        emb_list = emb.tolist()
        total_tokens = int(input_ids.size)

        return {
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": emb_list[i], "index": i}
                for i in range(len(emb_list))
            ],
            "model": info.model_name,
            "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
        }
