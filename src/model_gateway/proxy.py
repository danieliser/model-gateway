"""OpenAI-compatible proxy layer — translates gateway requests to backend calls."""

import logging
from typing import Any, Awaitable, Callable

import httpx
import litellm
from litellm import Router

from model_gateway.config import CLOUD_BACKENDS, TTS_BACKENDS, GatewayConfig

logger = logging.getLogger(__name__)


class ProxyManager:
    def __init__(
        self,
        config: GatewayConfig,
        ensure_model_fn: Callable[[str], Awaitable[int | None]] | None = None,
    ):
        self._config = config
        self._backend_urls: dict[str, str] = {}  # keyed by backend name or model alias
        self._router: Router | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._ensure_model_fn = ensure_model_fn

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

        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=120.0)

    def on_model_loaded(self, model_alias: str, backend_name: str, port: int) -> None:
        """Register a newly loaded model's port and rebuild router entry."""
        url = f"http://localhost:{port}/v1"
        self._backend_urls[model_alias] = url
        # Also register under backend name for backwards compat
        self._backend_urls[backend_name] = url

        # Rebuild router to include the new model
        model_cfg = self._config.models.get(model_alias)
        if model_cfg:
            entry = self._build_model_entry(model_alias, model_cfg)
            if entry:
                current = self._router.model_list if self._router else []
                # Remove existing entry for this alias, add updated one
                filtered = [e for e in current if e.get("model_name") != model_alias]
                filtered.append(entry)
                self._router = Router(model_list=filtered)

    def on_model_unloaded(self, model_alias: str, backend_name: str) -> None:
        """Remove a model's URL entry after unload."""
        self._backend_urls.pop(model_alias, None)
        # Only remove backend URL if no other model is using it
        # (another model on the same backend type may still be running)

        if self._router:
            filtered = [
                e for e in self._router.model_list
                if e.get("model_name") != model_alias
            ]
            if filtered:
                self._router = Router(model_list=filtered)
            else:
                self._router = None

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

        # Local backend — check model alias first, then backend name
        api_base = self._backend_urls.get(alias) or self._backend_urls.get(backend)
        if api_base is None:
            logger.warning("No port registered for model '%s' (backend '%s'), skipping", alias, backend)
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

    def register_external_url(self, backend_name: str, url: str) -> None:
        """Register a URL for an external backend (lm-studio, ollama, etc.)."""
        self._backend_urls[backend_name] = url

    def update_backend_url(self, backend_name: str, port: int) -> None:
        """Update URL for a local backend (called after model switch/restart)."""
        new_url = f"http://localhost:{port}/v1"
        self._backend_urls[backend_name] = new_url
        self.setup({backend_name: port})

    def _is_local_model(self, model: str) -> tuple[bool, str | None]:
        """Check if model routes to a local backend. Returns (is_local, api_base)."""
        model_cfg = self._config.models.get(model)
        if model_cfg is None:
            return False, None
        if model_cfg.backend in CLOUD_BACKENDS:
            return False, None
        # Check model alias first, then backend name
        api_base = self._backend_urls.get(model) or self._backend_urls.get(model_cfg.backend)
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

    async def embedding(
        self,
        model: str,
        input_text: str | list[str],
    ) -> Any:
        """Route an embedding request — pass-through for local, LiteLLM for cloud."""
        is_local, api_base = self._is_local_model(model)
        model_cfg = self._config.models.get(model)
        model_id = (model_cfg.model_id if model_cfg else model) or model

        if is_local and api_base and self._http_client:
            url = f"{api_base}/embeddings"
            return await self._http_client.post(
                url, json={"model": model_id, "input": input_text}
            )

        # Cloud backends go through LiteLLM Router
        if self._router is None:
            raise RuntimeError("ProxyManager not initialized — call setup() first")

        return await self._router.aembedding(model=model, input=input_text)

    async def audio_speech(
        self,
        model: str,
        text: str,
        voice: str = "af_heart",
        speed: float = 1.0,
        response_format: str = "wav",
        **kwargs: Any,
    ) -> bytes:
        """Route a TTS request to cloud or external backend. Returns audio bytes."""
        model_cfg = self._config.models.get(model)
        if model_cfg is None:
            raise RuntimeError(f"Unknown model: {model}")

        backend = model_cfg.backend

        if backend == "elevenlabs":
            return await self._elevenlabs_tts(model_cfg, text, voice, **kwargs)

        if backend == "google-tts":
            return await self._google_tts(model_cfg, text, voice, **kwargs)

        if backend == "tts-external":
            # Passthrough to external TTS server
            backend_cfg = self._config.backends.get(backend)
            host = (backend_cfg.host if backend_cfg and backend_cfg.host else None)
            if not host:
                raise RuntimeError(f"No host configured for tts-external backend")
            url = f"{host.rstrip('/')}/v1/audio/speech"
            payload: dict[str, Any] = {
                "model": model_cfg.model_id or model,
                "input": text,
                "voice": voice,
                "speed": speed,
                "response_format": response_format,
            }
            payload.update(kwargs)
            if not self._http_client:
                self._http_client = httpx.AsyncClient(timeout=120.0)
            resp = await self._http_client.post(url, json=payload)
            resp.raise_for_status()
            return resp.content

        raise RuntimeError(f"Unsupported TTS backend: {backend}")

    async def _elevenlabs_tts(
        self, model_cfg: Any, text: str, voice: str, **kwargs: Any
    ) -> bytes:
        """Call ElevenLabs TTS API."""
        import os
        api_key_env = model_cfg.api_key_env or (
            self._config.backends.get("elevenlabs", None)
            and self._config.backends["elevenlabs"].api_key_env
        ) or "ELEVENLABS_API_KEY"
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"ElevenLabs API key not set ({api_key_env})")

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}"
        headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
        payload = {
            "text": text,
            "model_id": model_cfg.model_id or "eleven_turbo_v2_5",
        }
        if not self._http_client:
            self._http_client = httpx.AsyncClient(timeout=120.0)
        resp = await self._http_client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.content

    async def _google_tts(
        self, model_cfg: Any, text: str, voice: str, **kwargs: Any
    ) -> bytes:
        """Call Google Cloud TTS API."""
        import os
        api_key_env = model_cfg.api_key_env or (
            self._config.backends.get("google-tts", None)
            and self._config.backends["google-tts"].api_key_env
        ) or "GOOGLE_CLOUD_API_KEY"
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Google TTS API key not set ({api_key_env})")

        url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={api_key}"
        payload = {
            "input": {"text": text},
            "voice": {
                "languageCode": kwargs.get("lang_code", "en-US"),
                "name": voice,
            },
            "audioConfig": {"audioEncoding": "LINEAR16"},
        }
        if not self._http_client:
            self._http_client = httpx.AsyncClient(timeout=120.0)
        resp = await self._http_client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        import base64
        return base64.b64decode(data["audioContent"])

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
                info["api_base"] = self._backend_urls.get(alias) or self._backend_urls.get(backend)
            result.append(info)
        return result
