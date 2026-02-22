"""OpenAI-compatible proxy layer — translates gateway requests to backend calls."""

import logging
from typing import Any, AsyncGenerator

import httpx
import litellm
from litellm import Router

from model_gateway.config import CLOUD_BACKENDS, GatewayConfig

logger = logging.getLogger(__name__)


class ProxyManager:
    def __init__(self, config: GatewayConfig):
        self._config = config
        self._backend_urls: dict[str, str] = {}
        self._router: Router | None = None
        self._http_client: httpx.AsyncClient | None = None

    def setup(self, backend_ports: dict[str, int]) -> None:
        """Configure LiteLLM Router with model aliases from our config."""
        for backend_name, port in backend_ports.items():
            self._backend_urls[backend_name] = f"http://localhost:{port}/v1"

        model_list = []
        for alias, model_cfg in self._config.models.items():
            entry = self._build_model_entry(alias, model_cfg)
            if entry is not None:
                model_list.append(entry)

        if model_list:
            self._router = Router(model_list=model_list)
        else:
            logger.warning("No models configured for LiteLLM Router")

        # Shared httpx client for local pass-through (connection pooling)
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=120.0)

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

        # Re-setup router with updated URLs
        self.setup({backend_name: port})

    def _is_local_model(self, model: str) -> tuple[bool, str | None]:
        """Check if model routes to a local backend. Returns (is_local, api_base)."""
        model_cfg = self._config.models.get(model)
        if model_cfg is None:
            return False, None
        if model_cfg.backend in CLOUD_BACKENDS:
            return False, None
        api_base = self._backend_urls.get(model_cfg.backend)
        return api_base is not None, api_base

    async def completion(
        self,
        model: str,
        messages: list[dict],
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Route a chat completion — pass-through for local, LiteLLM for cloud."""
        is_local, api_base = self._is_local_model(model)

        if is_local and api_base and self._http_client:
            return await self._local_passthrough(
                model, api_base, messages, stream, **kwargs,
            )

        # Cloud backends go through LiteLLM Router
        if self._router is None:
            raise RuntimeError("ProxyManager not initialized — call setup() first")

        response = await self._router.acompletion(
            model=model,
            messages=messages,
            stream=stream,
            **kwargs,
        )
        return response

    async def _local_passthrough(
        self,
        model: str,
        api_base: str,
        messages: list[dict],
        stream: bool,
        **kwargs: Any,
    ) -> httpx.Response:
        """Bypass LiteLLM — raw httpx proxy to local backend."""
        model_cfg = self._config.models[model]
        url = f"{api_base}/chat/completions"
        payload: dict[str, Any] = {
            "model": model_cfg.model_id or model,
            "messages": messages,
            "stream": stream,
        }
        payload.update(kwargs)

        if stream:
            req = self._http_client.build_request("POST", url, json=payload)
            return await self._http_client.send(req, stream=True)
        else:
            return await self._http_client.post(url, json=payload)

    async def close(self) -> None:
        """Close the httpx client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

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
            if backend not in CLOUD_BACKENDS:
                info["api_base"] = self._backend_urls.get(backend)
            result.append(info)
        return result
