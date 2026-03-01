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
from dataclasses import dataclass, field

import httpx

from model_gateway.config import CLOUD_BACKENDS, TTS_BACKENDS, BackendConfig, GatewayConfig, ModelConfig, get_log_dir
from model_gateway.embed_server import EmbedManager
from model_gateway.llm_server import LlmManager
from model_gateway.tts_server import TtsManager

logger = logging.getLogger(__name__)
_EXTERNAL_BACKENDS = {"ollama", "lm_studio", "lm-studio", "tts-external"}
_IN_PROCESS_BACKENDS = {"mlx-embed", "mlx", "mlx-audio"}


def _find_vllm_mlx() -> str | None:
    """Find vllm-mlx binary — check PATH first, then sibling of sys.executable."""
    found = shutil.which("vllm-mlx")
    if found:
        return found
    # When running as a daemon, PATH may not include the venv bin dir
    venv_bin = os.path.join(os.path.dirname(sys.executable), "vllm-mlx")
    if os.path.isfile(venv_bin) and os.access(venv_bin, os.X_OK):
        return venv_bin
    return None


def _is_mlx_available() -> bool:
    """Check macOS + arm64 + vllm-mlx binary available."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    return _find_vllm_mlx() is not None


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
    last_used: float | None = None
    error: str | None = None


@dataclass
class _RunningBackend:
    process: subprocess.Popen
    port: int
    pid: int
    model: str
    backend_name: str
    log_file: object  # file handle
    last_used: float = field(default_factory=time.monotonic)


class BackendManager:
    def __init__(self, config: GatewayConfig):
        self._config = config
        # Keyed by model alias (not backend name)
        self._running: dict[str, _RunningBackend] = {}
        self._status_errors: dict[str, str] = {}
        self._next_port: int = config.port + 1
        self._start_locks: dict[str, asyncio.Lock] = {}
        self._idle_monitor_task: asyncio.Task | None = None
        self._embed_manager = EmbedManager()
        self._llm_manager = LlmManager()
        self._tts_manager = TtsManager()

    def _alloc_port(self) -> int:
        """Allocate the next sequential port."""
        port = self._next_port
        self._next_port += 1
        return port

    @staticmethod
    def _is_port_free(port: int) -> bool:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return True
            except OSError:
                return False

    def _get_backend_cfg(self, backend_name: str) -> BackendConfig | None:
        return self._config.backends.get(backend_name)

    def _get_model_cfg(self, model_alias: str) -> ModelConfig | None:
        return self._config.models.get(model_alias)

    def _get_start_lock(self, model_alias: str) -> asyncio.Lock:
        if model_alias not in self._start_locks:
            self._start_locks[model_alias] = asyncio.Lock()
        return self._start_locks[model_alias]

    def _is_model_cached(self, model_id: str) -> bool:
        """Check if a model is available locally (HF cache or local path)."""
        expanded = os.path.expanduser(model_id)
        if os.path.isdir(expanded) and os.path.isfile(os.path.join(expanded, "config.json")):
            return True
        try:
            from huggingface_hub import try_to_load_from_cache
            result = try_to_load_from_cache(model_id, "config.json")
            return result is not None and isinstance(result, str)
        except ImportError:
            return False
        except Exception:
            return False

    def _build_command(
        self, backend_name: str, model_alias: str, model_cfg: ModelConfig, port: int
    ) -> list[str] | None:
        """Build the startup command for a local backend."""
        if backend_name == "mlx":
            model_id = model_cfg.model_id or model_cfg.model_path
            if not model_id:
                return None
            if not self._is_model_cached(model_id):
                logger.warning(
                    "Model '%s' (%s) not found locally. Download it first with: "
                    "huggingface-cli download %s",
                    model_alias, model_id, model_id,
                )
                self._status_errors[model_alias] = f"Model '{model_id}' not downloaded"
                return None
            vllm_bin = _find_vllm_mlx()
            if not vllm_bin:
                self._status_errors[model_alias] = "vllm-mlx binary not found"
                return None
            cmd = [
                vllm_bin, "serve", model_id,
                "--port", str(port),
                "--continuous-batching",
            ]
            return cmd
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
        if backend_name in ("mlx", "mlx-embed"):
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

    def _is_process_alive(self, model_alias: str) -> bool:
        """Check if the model's process is still running."""
        info = self._running.get(model_alias)
        if info is None:
            return False
        return info.process.poll() is None

    async def ensure_model(self, model_alias: str) -> int | None:
        """Ensure a model is loaded. Returns port if loaded, None on failure.

        This is the main entry point for lazy loading. If the model is already
        running, touches last_used and returns the port. Otherwise acquires a
        per-model lock to prevent duplicate starts.
        """
        # Fast path: already running and alive
        if model_alias in self._running and self._is_process_alive(model_alias):
            self._running[model_alias].last_used = time.monotonic()
            return self._running[model_alias].port

        model_cfg = self._get_model_cfg(model_alias)
        if model_cfg is None:
            self._status_errors[model_alias] = f"Unknown model alias: {model_alias}"
            return None

        backend_name = model_cfg.backend

        # Cloud / external backends don't have ports — return 0 as sentinel
        if backend_name in CLOUD_BACKENDS or backend_name in _EXTERNAL_BACKENDS:
            return 0

        # In-process backends — no subprocess, no port
        if backend_name in _IN_PROCESS_BACKENDS:
            if backend_name == "mlx-embed":
                manager = self._embed_manager
            elif backend_name == "mlx-audio":
                manager = self._tts_manager
            else:  # "mlx"
                manager = self._llm_manager

            if not manager.is_loaded(model_alias):
                model_id = model_cfg.model_id or model_cfg.model_path
                if not model_id:
                    self._status_errors[model_alias] = "No model_id or model_path"
                    return None
                try:
                    result = manager.load(model_alias, model_id)
                    # LlmManager.load() is async, EmbedManager.load() is sync
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as exc:
                    self._status_errors[model_alias] = str(exc)
                    logger.error("Failed to load in-process model %s: %s", model_alias, exc)
                    return None
            else:
                manager.touch(model_alias)
            return 0

        lock = self._get_start_lock(model_alias)
        async with lock:
            # Double-check after acquiring lock
            if model_alias in self._running and self._is_process_alive(model_alias):
                self._running[model_alias].last_used = time.monotonic()
                return self._running[model_alias].port

            # Clean up stale entry if present
            if model_alias in self._running:
                self._cleanup_stale(model_alias)

            started = await self._start_model(model_alias, backend_name, model_cfg)
            if started:
                return self._running[model_alias].port
            return None

    async def _start_model(
        self, model_alias: str, backend_name: str, model_cfg: ModelConfig
    ) -> bool:
        """Start a local backend process for the given model."""
        backend_cfg = self._get_backend_cfg(backend_name)
        port = self._alloc_port()
        cmd = self._build_command(backend_name, model_alias, model_cfg, port)
        if cmd is None:
            self._status_errors[model_alias] = f"Cannot build command for model '{model_alias}'"
            return False

        log_path = get_log_dir() / f"{model_alias}.log"

        for attempt in range(3):
            try:
                log_file = open(log_path, "ab")
                process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=log_file,
                    start_new_session=True,
                )
                self._running[model_alias] = _RunningBackend(
                    process=process,
                    port=port,
                    pid=process.pid,
                    model=model_alias,
                    backend_name=backend_name,
                    log_file=log_file,
                    last_used=time.monotonic(),
                )
                self._status_errors.pop(model_alias, None)

                if await self._wait_for_health(model_alias, backend_name, port, backend_cfg):
                    logger.info(
                        "Model %s started on %s (pid=%d, port=%d)",
                        model_alias, backend_name, process.pid, port,
                    )
                    return True

                if process.poll() is not None:
                    logger.warning(
                        "Model %s process exited (code=%s, attempt %d/3)",
                        model_alias, process.returncode, attempt + 1,
                    )
                else:
                    logger.warning(
                        "Model %s health check timed out (attempt %d/3)",
                        model_alias, attempt + 1,
                    )

                self._stop_model_sync(model_alias)
                await asyncio.sleep(2.0)

            except Exception as exc:
                logger.error("Error starting model %s: %s", model_alias, exc)
                self._status_errors[model_alias] = str(exc)
                self._running.pop(model_alias, None)

        self._status_errors[model_alias] = f"Model '{model_alias}' failed after 3 attempts"
        return False

    # Legacy compat — delegates to ensure_model
    async def start_backend(self, backend_name: str, model_alias: str) -> bool:
        """Start a local backend server for the given model. Returns True if started successfully."""
        port = await self.ensure_model(model_alias)
        return port is not None

    async def _wait_for_health(
        self, model_alias: str, backend_name: str, port: int,
        backend_cfg: BackendConfig | None, timeout: float = 120.0,
    ) -> bool:
        url = self._health_url(backend_name, port, backend_cfg)
        if not url:
            return True
        deadline = time.monotonic() + timeout
        async with httpx.AsyncClient() as client:
            while time.monotonic() < deadline:
                if not self._is_process_alive(model_alias):
                    return False
                try:
                    resp = await client.get(url, timeout=2.0)
                    if resp.status_code < 500:
                        return True
                except Exception:
                    pass
                await asyncio.sleep(2.0)
        return False

    def unload_model(self, model_alias: str) -> bool:
        """Stop a model's process and remove from _running (sync version).

        Returns True if unloaded, False if pinned or not running.
        For async LLM unloads, use unload_model_async() instead.
        """
        model_cfg = self._get_model_cfg(model_alias)
        if model_cfg and model_cfg.pin:
            logger.warning("Model %s is pinned — refusing to unload", model_alias)
            return False

        # In-process models
        if model_cfg and model_cfg.backend in _IN_PROCESS_BACKENDS:
            if model_cfg.backend == "mlx-embed":
                return self._embed_manager.unload(model_alias)
            if model_cfg.backend == "mlx-audio":
                return self._tts_manager.unload(model_alias)
            # "mlx" — schedule async unload, return True optimistically
            if self._llm_manager.is_loaded(model_alias):
                asyncio.ensure_future(self._llm_manager.unload(model_alias))
                return True
            return False

        if model_alias not in self._running:
            return False

        self._stop_model_sync(model_alias)
        return True

    async def unload_model_async(self, model_alias: str) -> bool:
        """Async version of unload_model — properly awaits LLM engine shutdown."""
        model_cfg = self._get_model_cfg(model_alias)
        if model_cfg and model_cfg.pin:
            logger.warning("Model %s is pinned — refusing to unload", model_alias)
            return False

        if model_cfg and model_cfg.backend in _IN_PROCESS_BACKENDS:
            if model_cfg.backend == "mlx-embed":
                return self._embed_manager.unload(model_alias)
            if model_cfg.backend == "mlx-audio":
                return self._tts_manager.unload(model_alias)
            return await self._llm_manager.unload(model_alias)

        if model_alias not in self._running:
            return False

        self._stop_model_sync(model_alias)
        return True

    def _cleanup_stale(self, model_alias: str) -> None:
        """Remove a stale entry for a dead process."""
        info = self._running.pop(model_alias, None)
        if info:
            try:
                info.log_file.close()
            except Exception:
                pass

    def _stop_model_sync(self, model_alias: str) -> None:
        """Stop a model's process synchronously."""
        info = self._running.pop(model_alias, None)
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

        logger.info("Model %s stopped", model_alias)

    # Legacy compat names
    def _stop_backend_sync(self, backend_name: str) -> None:
        """Legacy: stop by backend name. Finds the model alias and delegates."""
        # Find model alias running on this backend
        for alias, info in list(self._running.items()):
            if info.backend_name == backend_name:
                self._stop_model_sync(alias)
                return

    async def stop_backend(self, backend_name: str) -> None:
        """Graceful shutdown: SIGTERM, wait 5s, SIGKILL if needed."""
        self._stop_backend_sync(backend_name)

    async def stop_model(self, model_alias: str) -> None:
        """Async wrapper for stopping a model by alias."""
        self._stop_model_sync(model_alias)

    async def health_check(self, backend_name: str) -> bool:
        """HTTP health probe to local server. Returns True if healthy."""
        backend_cfg = self._get_backend_cfg(backend_name)

        # Cloud backends: check env var
        if backend_name in CLOUD_BACKENDS:
            if backend_cfg and backend_cfg.api_key_env:
                return os.environ.get(backend_cfg.api_key_env) is not None
            return False

        # Find a running model on this backend
        port = None
        model_alias = None
        for alias, info in self._running.items():
            if info.backend_name == backend_name:
                model_alias = alias
                if not self._is_process_alive(alias):
                    self._cleanup_stale(alias)
                    return False
                port = info.port
                break

        url = self._health_url(backend_name, port, backend_cfg)
        if not url:
            return False

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=5.0)
                return resp.status_code < 500
        except Exception:
            return False

    async def switch_model(self, old_model: str, new_model: str) -> bool:
        """Unload old model, load new one. Returns True on success."""
        self.unload_model(old_model)
        port = await self.ensure_model(new_model)
        return port is not None

    # --- Idle monitor ---

    def start_idle_monitor(self) -> None:
        """Launch the background idle sweep task."""
        if self._idle_monitor_task is not None:
            return
        self._idle_monitor_task = asyncio.create_task(self._idle_sweep_loop())

    def stop_idle_monitor(self) -> None:
        """Cancel the idle monitor task."""
        if self._idle_monitor_task is not None:
            self._idle_monitor_task.cancel()
            self._idle_monitor_task = None

    async def _idle_sweep_loop(self) -> None:
        """Background loop that periodically unloads idle models."""
        interval = self._config.idle_check_interval
        try:
            while True:
                await asyncio.sleep(interval)
                self._idle_sweep()
        except asyncio.CancelledError:
            pass

    def _idle_sweep(self) -> None:
        """Walk _running, unload models that have been idle past their timeout."""
        now = time.monotonic()
        global_timeout = self._config.idle_timeout

        for alias in list(self._running.keys()):
            info = self._running.get(alias)
            if info is None:
                continue

            # Clean up dead processes
            if not self._is_process_alive(alias):
                self._cleanup_stale(alias)
                continue

            # Check pin
            model_cfg = self._get_model_cfg(alias)
            if model_cfg and model_cfg.pin:
                continue

            # Per-model timeout overrides global
            timeout = (model_cfg.idle_timeout if model_cfg and model_cfg.idle_timeout is not None else global_timeout)
            if now - info.last_used > timeout:
                logger.info("Model %s idle for >%ds — unloading", alias, timeout)
                self._stop_model_sync(alias)

        # In-process models
        for alias, model_cfg in self._config.models.items():
            if model_cfg.backend not in _IN_PROCESS_BACKENDS:
                continue
            if model_cfg.pin:
                continue

            # Pick the right manager
            if model_cfg.backend == "mlx-embed":
                manager = self._embed_manager
            elif model_cfg.backend == "mlx-audio":
                manager = self._tts_manager
            else:
                manager = self._llm_manager

            if not manager.is_loaded(alias):
                continue
            last_used = manager.get_last_used(alias)
            if last_used is None:
                continue
            timeout = (model_cfg.idle_timeout if model_cfg.idle_timeout is not None else global_timeout)
            if now - last_used > timeout:
                logger.info("In-process model %s idle for >%ds — unloading", alias, timeout)
                result = manager.unload(alias)
                # LlmManager.unload() is async
                if asyncio.iscoroutine(result):
                    asyncio.ensure_future(result)

    def get_status(self) -> dict[str, BackendStatus]:
        """Return status of all models (loaded and available)."""
        result: dict[str, BackendStatus] = {}

        # Running models
        for alias, info in list(self._running.items()):
            if self._is_process_alive(alias):
                result[alias] = BackendStatus(
                    name=alias,
                    running=True,
                    port=info.port,
                    pid=info.pid,
                    model_alias=alias,
                    last_used=info.last_used,
                )
            else:
                self._cleanup_stale(alias)

        # In-process models
        for alias, model_cfg in self._config.models.items():
            if alias in result:
                continue
            if model_cfg.backend not in _IN_PROCESS_BACKENDS:
                continue
            if model_cfg.backend == "mlx-embed":
                manager = self._embed_manager
            elif model_cfg.backend == "mlx-audio":
                manager = self._tts_manager
            else:
                manager = self._llm_manager
            if manager.is_loaded(alias):
                result[alias] = BackendStatus(
                    name=alias,
                    running=True,
                    model_alias=alias,
                    last_used=manager.get_last_used(alias),
                )

        # Configured-but-unloaded models
        for alias, model_cfg in self._config.models.items():
            if alias in result:
                continue
            backend_name = model_cfg.backend
            # External backends — assume running if enabled with host
            if backend_name in _EXTERNAL_BACKENDS:
                cfg = self._config.backends.get(backend_name)
                if cfg and cfg.enabled and cfg.host:
                    result[alias] = BackendStatus(name=alias, running=True, model_alias=alias)
                    continue
            # Cloud backends — always available
            if backend_name in CLOUD_BACKENDS:
                result[alias] = BackendStatus(name=alias, running=True, model_alias=alias)
                continue
            error = self._status_errors.get(alias)
            result[alias] = BackendStatus(
                name=alias,
                running=False,
                model_alias=alias,
                error=error,
            )

        return result

    def get_port(self, model_alias: str) -> int | None:
        """Return the port for a running model, or None if not loaded."""
        info = self._running.get(model_alias)
        if info and self._is_process_alive(model_alias):
            return info.port
        return None

    def is_model_loaded(self, model_alias: str) -> bool:
        """Check if a model is loaded (subprocess or in-process)."""
        if self.get_port(model_alias) is not None:
            return True
        model_cfg = self._get_model_cfg(model_alias)
        if model_cfg and model_cfg.backend in _IN_PROCESS_BACKENDS:
            if model_cfg.backend == "mlx-embed":
                return self._embed_manager.is_loaded(model_alias)
            if model_cfg.backend == "mlx-audio":
                return self._tts_manager.is_loaded(model_alias)
            return self._llm_manager.is_loaded(model_alias)
        if model_cfg and model_cfg.backend in CLOUD_BACKENDS:
            return True
        return False

    def get_backend_name(self, model_alias: str) -> str | None:
        """Return the backend name for a model alias from config."""
        model_cfg = self._get_model_cfg(model_alias)
        return model_cfg.backend if model_cfg else None

    async def cleanup(self) -> None:
        """Stop all running backends. Called on gateway shutdown."""
        self.stop_idle_monitor()
        # Stop subprocess backends
        aliases = list(self._running.keys())
        for alias in aliases:
            self._stop_model_sync(alias)
        # Stop in-process LLM engines
        for alias in list(self._models_to_unload_llm()):
            await self._llm_manager.unload(alias)
        # Unload in-process embeddings
        for alias, cfg in self._config.models.items():
            if cfg.backend == "mlx-embed" and self._embed_manager.is_loaded(alias):
                self._embed_manager.unload(alias)
        # Unload in-process TTS models
        for alias, cfg in self._config.models.items():
            if cfg.backend == "mlx-audio" and self._tts_manager.is_loaded(alias):
                self._tts_manager.unload(alias)

    def _models_to_unload_llm(self) -> list[str]:
        """Return list of loaded LLM model aliases."""
        return [
            alias for alias, cfg in self._config.models.items()
            if cfg.backend == "mlx" and self._llm_manager.is_loaded(alias)
        ]
