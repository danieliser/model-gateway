"""Backend adapters for local and remote model providers (mlx, ollama, anthropic, openai, etc.)."""

import asyncio
import logging
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass

import httpx

from model_gateway.config import CLOUD_BACKENDS, BackendConfig, GatewayConfig, ModelConfig, get_log_dir

logger = logging.getLogger(__name__)
_EXTERNAL_BACKENDS = {"ollama", "lm_studio", "lm-studio"}


def _is_mlx_available() -> bool:
    """Check macOS + arm64 + vllm-mlx binary available."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    return shutil.which("vllm-mlx") is not None


def _is_binary_available(name: str) -> bool:
    """Check if binary exists in PATH."""
    return shutil.which(name) is not None


@dataclass
class BackendStatus:
    name: str
    running: bool
    port: int | None = None
    pid: int | None = None
    model_alias: str | None = None
    error: str | None = None


@dataclass
class _RunningBackend:
    process: subprocess.Popen
    port: int
    pid: int
    model: str
    log_file: object  # file handle


class BackendManager:
    def __init__(self, config: GatewayConfig):
        self._config = config
        self._running: dict[str, _RunningBackend] = {}
        self._status_errors: dict[str, str] = {}
        self._next_port: int = 8801

    def _alloc_port(self) -> int:
        port = self._next_port
        self._next_port += 1
        return port

    def _get_backend_cfg(self, backend_name: str) -> BackendConfig | None:
        return self._config.backends.get(backend_name)

    def _get_model_cfg(self, model_alias: str) -> ModelConfig | None:
        return self._config.models.get(model_alias)

    def _is_model_cached(self, model_id: str) -> bool:
        """Check if a HuggingFace model is already downloaded locally."""
        try:
            from huggingface_hub import try_to_load_from_cache
            # Check for config.json as a proxy — every model has one
            result = try_to_load_from_cache(model_id, "config.json")
            return result is not None and isinstance(result, str)
        except ImportError:
            # If huggingface_hub isn't available, assume not cached
            return False
        except Exception:
            return False

    def _build_command(
        self, backend_name: str, model_cfg: ModelConfig, port: int
    ) -> list[str] | None:
        """Build the startup command for a local backend."""
        if backend_name == "mlx":
            model_id = model_cfg.model_id or model_cfg.model_path
            if not model_id:
                return None
            if not self._is_model_cached(model_id):
                logger.warning(
                    "Model '%s' not found locally. Download it first with: "
                    "huggingface-cli download %s",
                    model_id, model_id,
                )
                self._status_errors[backend_name] = f"Model '{model_id}' not downloaded"
                return None
            return [
                "vllm-mlx", "serve", model_id,
                "--port", str(port),
                "--continuous-batching",
            ]
        elif backend_name in ("llama.cpp", "llama-cpp", "llamacpp"):
            model_path = model_cfg.model_path or model_cfg.model_id
            if not model_path:
                return None
            backend_cfg = self._get_backend_cfg(backend_name)
            binary_name = (backend_cfg.binary if backend_cfg and backend_cfg.binary else "llama-server")
            return [binary_name, "-m", model_path, "--port", str(port)]
        return None

    def _health_url(self, backend_name: str, port: int | None, backend_cfg: BackendConfig | None) -> str:
        """Return the health check URL for a backend."""
        if backend_name == "mlx":
            return f"http://localhost:{port}/health"
        elif backend_name in ("llama.cpp", "llama-cpp", "llamacpp"):
            return f"http://localhost:{port}/health"
        elif backend_name == "ollama":
            host = (backend_cfg.host if backend_cfg and backend_cfg.host else "http://localhost:11434")
            return f"{host}/api/tags"
        elif backend_name in ("lm_studio", "lm-studio"):
            host = (backend_cfg.host if backend_cfg and backend_cfg.host else "http://localhost:1234")
            return f"{host}/v1/models"
        return ""

    def _is_process_alive(self, backend_name: str) -> bool:
        """Check if the backend's process is still running."""
        info = self._running.get(backend_name)
        if info is None:
            return False
        return info.process.poll() is None

    async def start_backend(self, backend_name: str, model_alias: str) -> bool:
        """Start a local backend server for the given model. Returns True if started successfully."""
        if backend_name in self._running:
            if self._is_process_alive(backend_name):
                logger.debug("Backend %s already running", backend_name)
                return True
            # Process died — clean up stale entry
            self._cleanup_stale(backend_name)

        backend_cfg = self._get_backend_cfg(backend_name)
        model_cfg = self._get_model_cfg(model_alias)

        if model_cfg is None:
            self._status_errors[backend_name] = f"Unknown model alias: {model_alias}"
            return False

        # External / cloud backends don't need subprocess
        if backend_name in CLOUD_BACKENDS or backend_name in _EXTERNAL_BACKENDS:
            return True

        cmd = self._build_command(backend_name, model_cfg, self._alloc_port())
        if cmd is None:
            self._status_errors[backend_name] = f"Cannot build command for backend '{backend_name}'"
            return False

        log_path = get_log_dir() / f"{backend_name}.log"

        for attempt in range(3):
            port = self._running[backend_name].port if backend_name in self._running else self._next_port - 1
            try:
                log_file = open(log_path, "ab")
                # Use subprocess.Popen with start_new_session to fully detach
                # from uvicorn's process group and signal handling.
                process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=log_file,
                    start_new_session=True,
                )
                self._running[backend_name] = _RunningBackend(
                    process=process,
                    port=port,
                    pid=process.pid,
                    model=model_alias,
                    log_file=log_file,
                )
                self._status_errors.pop(backend_name, None)

                # Wait for health
                if await self._wait_for_health(backend_name, port, backend_cfg):
                    logger.info("Backend %s started (pid=%d, port=%d)", backend_name, process.pid, port)
                    return True

                # Check if process crashed vs just slow
                if process.poll() is not None:
                    logger.warning("Backend %s process exited (code=%s, attempt %d/3)", backend_name, process.returncode, attempt + 1)
                else:
                    logger.warning("Backend %s health check timed out (attempt %d/3)", backend_name, attempt + 1)

                self._stop_backend_sync(backend_name)

                # Wait for port release before retry
                await asyncio.sleep(2.0)

            except Exception as exc:
                logger.error("Error starting backend %s: %s", backend_name, exc)
                self._status_errors[backend_name] = str(exc)
                self._running.pop(backend_name, None)

        self._status_errors[backend_name] = f"Backend '{backend_name}' failed after 3 attempts"
        return False

    async def _wait_for_health(
        self, backend_name: str, port: int, backend_cfg: BackendConfig | None, timeout: float = 120.0
    ) -> bool:
        url = self._health_url(backend_name, port, backend_cfg)
        if not url:
            return True  # no health check available
        deadline = time.monotonic() + timeout
        async with httpx.AsyncClient() as client:
            while time.monotonic() < deadline:
                # Check if process died while we're waiting
                if not self._is_process_alive(backend_name):
                    return False
                try:
                    resp = await client.get(url, timeout=2.0)
                    if resp.status_code < 500:
                        return True
                except Exception:
                    pass
                await asyncio.sleep(2.0)
        return False

    def _cleanup_stale(self, backend_name: str) -> None:
        """Remove a stale entry for a dead process."""
        info = self._running.pop(backend_name, None)
        if info:
            try:
                info.log_file.close()
            except Exception:
                pass

    def _stop_backend_sync(self, backend_name: str) -> None:
        """Stop a backend process synchronously."""
        info = self._running.pop(backend_name, None)
        if info is None:
            return

        proc = info.process
        if proc.poll() is None:
            try:
                os.kill(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                try:
                    os.kill(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=2.0)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass

        try:
            info.log_file.close()
        except Exception:
            pass

        logger.info("Backend %s stopped", backend_name)

    async def stop_backend(self, backend_name: str) -> None:
        """Graceful shutdown: SIGTERM, wait 5s, SIGKILL if needed."""
        self._stop_backend_sync(backend_name)

    async def health_check(self, backend_name: str) -> bool:
        """HTTP health probe to local server. Returns True if healthy."""
        backend_cfg = self._get_backend_cfg(backend_name)

        # Cloud backends: check env var
        if backend_name in CLOUD_BACKENDS:
            if backend_cfg and backend_cfg.api_key_env:
                return os.environ.get(backend_cfg.api_key_env) is not None
            return False

        # Running subprocess backends
        port = None
        if backend_name in self._running:
            if not self._is_process_alive(backend_name):
                self._cleanup_stale(backend_name)
                return False
            port = self._running[backend_name].port

        url = self._health_url(backend_name, port, backend_cfg)
        if not url:
            return False

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=5.0)
                return resp.status_code < 500
        except Exception:
            return False

    async def switch_model(self, backend_name: str, model_alias: str) -> bool:
        """Stop current model on backend, start new one. Returns True on success."""
        await self.stop_backend(backend_name)
        return await self.start_backend(backend_name, model_alias)

    def get_status(self) -> dict[str, BackendStatus]:
        """Return status of all backends."""
        result: dict[str, BackendStatus] = {}

        for name, info in list(self._running.items()):
            if self._is_process_alive(name):
                result[name] = BackendStatus(
                    name=name,
                    running=True,
                    port=info.port,
                    pid=info.pid,
                    model_alias=info.model,
                )
            else:
                self._cleanup_stale(name)

        for name in self._config.backends:
            if name not in result:
                error = self._status_errors.get(name)
                result[name] = BackendStatus(
                    name=name,
                    running=False,
                    error=error,
                )

        return result

    async def cleanup(self) -> None:
        """Stop all running backends. Called on gateway shutdown."""
        names = list(self._running.keys())
        for name in names:
            await self.stop_backend(name)
