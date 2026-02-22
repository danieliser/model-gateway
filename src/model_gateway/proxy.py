"""OpenAI-compatible proxy layer — translates gateway requests to backend calls."""

import logging
from typing import Any, AsyncGenerator

import litellm

from model_gateway.config import GatewayConfig

logger = logging.getLogger(__name__)

_CLOUD_BACKENDS = {"anthropic", "openai"}


class ProxyManager:
    def __init__(self, config: GatewayConfig):
        """Configure LiteLLM with model aliases from our config."""
        self._config = config
        # Track current backend URLs so we can update them without full re-setup
        self._backend_urls: dict[str, str] = {}

    def setup(self, backend_ports: dict[str, int]) -> None:
        """Configure LiteLLM model routing.

        - For local backends (mlx, llama-cpp): register as OpenAI-compatible
          at http://localhost:<port>/v1
        - For cloud backends (anthropic): use litellm's native provider format
        - For cloud backends (openai): use litellm's native format

        Sets litellm.model_list with our model aliases.
        """
        # Store resolved URLs for local backends
        for backend_name, port in backend_ports.items():
            self._backend_urls[backend_name] = f"http://localhost:{port}/v1"

        model_list = []
        for alias, model_cfg in self._config.models.items():
            entry = self._build_model_entry(alias, model_cfg)
            if entry is not None:
                model_list.append(entry)

        litellm.model_list = model_list

    def _build_model_entry(self, alias: str, model_cfg: Any) -> dict | None:
        """Build a single LiteLLM model_list entry."""
        backend = model_cfg.backend

        if backend == "anthropic":
            model_id = model_cfg.model_id or alias
            api_key_env = model_cfg.api_key_env or (
                self._config.backends.get("anthropic", None)
                and self._config.backends["anthropic"].api_key_env
            ) or "ANTHROPIC_API_KEY"
            return {
                "model_name": alias,
                "litellm_params": {
                    "model": f"anthropic/{model_id}",
                    "api_key": f"os.environ/{api_key_env}",
                },
            }

        if backend == "openai":
            model_id = model_cfg.model_id or alias
            api_key_env = model_cfg.api_key_env or (
                self._config.backends.get("openai", None)
                and self._config.backends["openai"].api_key_env
            ) or "OPENAI_API_KEY"
            return {
                "model_name": alias,
                "litellm_params": {
                    "model": model_id,
                    "api_key": f"os.environ/{api_key_env}",
                },
            }

        # Local backend — must have a port registered
        api_base = self._backend_urls.get(backend)
        if api_base is None:
            logger.warning("No port registered for local backend '%s', skipping model '%s'", backend, alias)
            return None

        model_id = model_cfg.model_id or alias
        return {
            "model_name": alias,
            "litellm_params": {
                "model": f"openai/{model_id}",
                "api_base": api_base,
                "api_key": "not-needed",
            },
        }

    def update_backend_url(self, backend_name: str, port: int) -> None:
        """Update URL for a local backend (called after model switch/restart)."""
        new_url = f"http://localhost:{port}/v1"
        self._backend_urls[backend_name] = new_url

        # Update existing entries in litellm.model_list
        for entry in litellm.model_list:
            params = entry.get("litellm_params", {})
            if params.get("api_base", "").startswith("http://localhost:"):
                alias = entry["model_name"]
                model_cfg = self._config.models.get(alias)
                if model_cfg and model_cfg.backend == backend_name:
                    params["api_base"] = new_url

    async def completion(
        self,
        model: str,
        messages: list[dict],
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Route a chat completion through LiteLLM.

        Uses litellm.acompletion() for async.
        For streaming: returns async generator.
        For non-streaming: returns completion response.
        """
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            stream=stream,
            **kwargs,
        )
        return response

    def get_available_models(self) -> list[dict]:
        """Return list of configured models with their backend info."""
        result = []
        for alias, model_cfg in self._config.models.items():
            backend = model_cfg.backend
            info: dict[str, Any] = {
                "alias": alias,
                "backend": backend,
                "model_id": model_cfg.model_id,
            }
            if backend not in _CLOUD_BACKENDS:
                info["api_base"] = self._backend_urls.get(backend)
            result.append(info)
        return result
