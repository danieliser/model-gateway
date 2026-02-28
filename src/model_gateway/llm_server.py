"""In-process MLX chat inference using vllm-mlx EngineCore.

Manages model loading/unloading and provides generate()/stream_generate()
that return OpenAI-compatible responses. No subprocess, no extra port —
runs inside the gateway process with continuous batching.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Lazy imports — resolved on first use, patchable at module level for tests
mlx_load = None
EngineCore = None
EngineConfig = None
SamplingParams = None


def _ensure_imports() -> None:
    """Import MLX/vllm-mlx on first use (avoids import errors on non-Apple)."""
    global mlx_load, EngineCore, EngineConfig, SamplingParams
    if mlx_load is not None:
        return
    from mlx_lm import load
    from vllm_mlx import EngineCore as _EC, EngineConfig as _ECfg, SamplingParams as _SP
    mlx_load = load
    EngineCore = _EC
    EngineConfig = _ECfg
    SamplingParams = _SP


@dataclass
class _LoadedLlmModel:
    model: Any
    tokenizer: Any
    engine: Any  # vllm_mlx.EngineCore
    model_name: str
    last_used: float = field(default_factory=time.monotonic)


class LlmManager:
    """Manages in-process MLX chat models via vllm-mlx EngineCore."""

    def __init__(self) -> None:
        self._models: dict[str, _LoadedLlmModel] = {}

    def is_loaded(self, model_alias: str) -> bool:
        return model_alias in self._models

    async def load(self, model_alias: str, model_path: str) -> None:
        """Load a chat model into memory and start its EngineCore."""
        if model_alias in self._models:
            self._models[model_alias].last_used = time.monotonic()
            return

        resolved = os.path.expanduser(model_path)
        logger.info("Loading LLM '%s' from %s", model_alias, resolved)

        _ensure_imports()
        model, tokenizer = mlx_load(resolved)

        config = EngineConfig(stream_interval=1)
        engine = EngineCore(model, tokenizer, config)
        await engine.start()

        self._models[model_alias] = _LoadedLlmModel(
            model=model,
            tokenizer=tokenizer,
            engine=engine,
            model_name=os.path.basename(resolved.rstrip("/")),
        )
        logger.info("LLM '%s' loaded with EngineCore", model_alias)

    async def unload(self, model_alias: str) -> bool:
        """Stop engine and unload model from memory. Returns True if unloaded."""
        info = self._models.pop(model_alias, None)
        if info is None:
            return False

        try:
            await info.engine.stop()
            info.engine.close()
        except Exception as exc:
            logger.warning("Error stopping engine for '%s': %s", model_alias, exc)

        del info.model
        del info.tokenizer
        logger.info("LLM '%s' unloaded", model_alias)
        return True

    def touch(self, model_alias: str) -> None:
        """Update last_used timestamp."""
        info = self._models.get(model_alias)
        if info:
            info.last_used = time.monotonic()

    def get_last_used(self, model_alias: str) -> float | None:
        info = self._models.get(model_alias)
        return info.last_used if info else None

    def _format_prompt(self, info: _LoadedLlmModel, messages: list[dict]) -> str:
        """Apply chat template to format messages into a prompt string."""
        return info.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    def _strip_stop_tokens(self, info: _LoadedLlmModel, text: str) -> str:
        """Strip EOS / chat-template stop tokens from generated text."""
        # Collect known stop strings from tokenizer
        stop_strings: list[str] = []
        eos = getattr(info.tokenizer, "eos_token", None)
        if isinstance(eos, str) and eos:
            stop_strings.append(eos)
        # Many chat models define additional stop tokens (e.g. <|im_end|>)
        extra = getattr(info.tokenizer, "additional_special_tokens", None)
        if isinstance(extra, list):
            stop_strings.extend(t for t in extra if isinstance(t, str))
        # Also check chat_template stop tokens if present
        if hasattr(info.tokenizer, "chat_template") and info.tokenizer.chat_template:
            for candidate in ("<|im_end|>", "<|eot_id|>", "</s>", "<|end|>"):
                if candidate in info.tokenizer.chat_template and candidate not in stop_strings:
                    stop_strings.append(candidate)

        for s in stop_strings:
            if text.endswith(s):
                text = text[: -len(s)]
        return text.rstrip()

    def _build_sampling_params(self, **kwargs: Any) -> Any:
        """Build SamplingParams from request kwargs."""
        _ensure_imports()

        param_kwargs: dict[str, Any] = {}
        if "max_tokens" in kwargs:
            param_kwargs["max_tokens"] = kwargs["max_tokens"]
        if "temperature" in kwargs:
            param_kwargs["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            param_kwargs["top_p"] = kwargs["top_p"]
        if "repetition_penalty" in kwargs:
            param_kwargs["repetition_penalty"] = kwargs["repetition_penalty"]
        if "stop" in kwargs and kwargs["stop"] is not None:
            stop = kwargs["stop"]
            param_kwargs["stop"] = stop if isinstance(stop, list) else [stop]

        return SamplingParams(**param_kwargs)

    async def generate(
        self, model_alias: str, messages: list[dict], **kwargs: Any
    ) -> dict:
        """Generate a chat completion. Returns an OpenAI-compatible response dict."""
        info = self._models.get(model_alias)
        if info is None:
            raise RuntimeError(f"LLM '{model_alias}' not loaded")

        info.last_used = time.monotonic()
        prompt = self._format_prompt(info, messages)
        sampling_params = self._build_sampling_params(**kwargs)

        request_id = await info.engine.add_request(prompt, sampling_params)

        # Collect full output
        final_output = None
        async for output in info.engine.stream_outputs(request_id):
            if output.finished:
                final_output = output
                break
            final_output = output

        if final_output is None:
            raise RuntimeError(f"No output from engine for '{model_alias}'")

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": info.model_name,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": self._strip_stop_tokens(info, final_output.output_text),
                },
                "finish_reason": final_output.finish_reason or "stop",
            }],
            "usage": {
                "prompt_tokens": final_output.prompt_tokens,
                "completion_tokens": final_output.completion_tokens,
                "total_tokens": final_output.prompt_tokens + final_output.completion_tokens,
            },
        }

    async def stream_generate(
        self, model_alias: str, messages: list[dict], **kwargs: Any
    ) -> AsyncGenerator[dict, None]:
        """Stream a chat completion. Yields OpenAI-compatible SSE chunk dicts."""
        info = self._models.get(model_alias)
        if info is None:
            raise RuntimeError(f"LLM '{model_alias}' not loaded")

        info.last_used = time.monotonic()
        prompt = self._format_prompt(info, messages)
        sampling_params = self._build_sampling_params(**kwargs)

        request_id = await info.engine.add_request(prompt, sampling_params)
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        async for output in info.engine.stream_outputs(request_id):
            if output.finished:
                # Final chunk with finish_reason
                yield {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": info.model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": output.finish_reason or "stop",
                    }],
                }
                break
            else:
                yield {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": info.model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": self._strip_stop_tokens(info, output.new_text)},
                        "finish_reason": None,
                    }],
                }
