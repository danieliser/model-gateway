"""FastAPI/uvicorn server setup and lifecycle management for the gateway."""

import json
import subprocess
import time
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from model_gateway.backends import BackendManager
from model_gateway.config import GatewayConfig, load_config, validate_config
from model_gateway.proxy import ProxyManager
from model_gateway.routing import TaskRouter

app = FastAPI(title="Model Gateway", version="0.1.0")

# Global state (initialized on startup)
_config: GatewayConfig | None = None
_backend_manager: BackendManager | None = None
_proxy_manager: ProxyManager | None = None
_task_router: TaskRouter | None = None


@app.on_event("startup")
async def startup() -> None:
    """Load config, init managers, start default backend."""
    global _config, _backend_manager, _proxy_manager, _task_router

    _config = load_config()
    errors = validate_config(_config)
    if any(e.startswith("ERROR") for e in errors):
        raise RuntimeError(f"Config errors: {errors}")

    _backend_manager = BackendManager(_config)
    _proxy_manager = ProxyManager(_config)
    _task_router = TaskRouter(_config)

    # Start default model's backend if local
    if _config.default_model and _config.default_model in _config.models:
        model = _config.models[_config.default_model]
        if model.backend not in ("anthropic", "openai"):
            started = await _backend_manager.start_backend(model.backend, _config.default_model)
            if started:
                port = _backend_manager.get_status()[model.backend].port
                if port:
                    _proxy_manager.setup({model.backend: port})
            else:
                import logging
                logging.getLogger(__name__).warning(
                    "Default backend '%s' failed to start — gateway will serve cloud models only",
                    model.backend,
                )

    # Setup proxy for cloud backends (and any already-running local ones)
    _proxy_manager.setup(_get_backend_ports())


@app.on_event("shutdown")
async def shutdown() -> None:
    if _backend_manager:
        await _backend_manager.cleanup()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request) -> StreamingResponse | JSONResponse:
    if not _config or not _backend_manager or not _proxy_manager or not _task_router:
        raise HTTPException(503, "Gateway not initialized yet")

    body = await request.json()
    model = body.get("model", _config.default_model)
    messages = body.get("messages", [])
    stream = body.get("stream", False)

    # Resolve model through task router
    backend_health = {
        name: status.running
        for name, status in _backend_manager.get_status().items()
    }
    try:
        resolved_model = _task_router.resolve_model(
            model, dict(request.headers), backend_health
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # Call through proxy
    if stream:
        async def generate():
            async for chunk in await _proxy_manager.completion(
                resolved_model, messages, stream=True
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        result = await _proxy_manager.completion(resolved_model, messages)
        # LiteLLM returns ModelResponse objects — convert to dict
        content = result.model_dump() if hasattr(result, "model_dump") else result
        return JSONResponse(content=content)


@app.get("/v1/models")
async def list_models() -> dict:
    models = _proxy_manager.get_available_models()
    return {"object": "list", "data": models}


@app.get("/health")
async def health() -> dict:
    statuses = _backend_manager.get_status()
    return {
        "status": "healthy",
        "port": _config.port,
        "default_model": _config.default_model,
        "backends": {
            name: {"running": s.running, "port": s.port, "model": s.model_alias}
            for name, s in statuses.items()
        },
    }


@app.get("/gateway/config")
async def get_config() -> dict:
    return _config_to_safe_dict(_config)


@app.post("/gateway/switch")
async def switch_model(request: Request) -> dict:
    body = await request.json()
    model_alias = body.get("model")
    if model_alias not in _config.models:
        raise HTTPException(404, f"Unknown model: {model_alias}")
    model_cfg = _config.models[model_alias]
    success = await _backend_manager.switch_model(model_cfg.backend, model_alias)
    if success:
        port = _backend_manager.get_status()[model_cfg.backend].port
        _proxy_manager.update_backend_url(model_cfg.backend, port)
    return {"success": success, "model": model_alias}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_backend_ports() -> dict[str, int]:
    """Get current port mapping from backend manager."""
    return {
        name: status.port
        for name, status in _backend_manager.get_status().items()
        if status.port is not None
    }


def _config_to_safe_dict(config: GatewayConfig) -> dict:
    """Recursively convert config to dict, redacting api_key values."""
    raw = {
        "port": config.port,
        "default_model": config.default_model,
        "models": {
            name: {
                "backend": m.backend,
                "model_id": m.model_id,
                "model_path": m.model_path,
                "api_key_env": m.api_key_env,
            }
            for name, m in config.models.items()
        },
        "backends": {
            name: {
                "enabled": b.enabled,
                "host": b.host,
                "binary": b.binary,
                "api_key_env": b.api_key_env,
            }
            for name, b in config.backends.items()
        },
        "task_routing": config.task_routing,
        "fallback_chain": config.fallback_chain,
    }
    return _redact_api_keys(raw)


def _redact_api_keys(obj: Any) -> Any:
    """Recursively replace values for keys containing 'api_key' with '***'."""
    if isinstance(obj, dict):
        return {
            k: "***" if "api_key" in k else _redact_api_keys(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_redact_api_keys(item) for item in obj]
    return obj


# ---------------------------------------------------------------------------
# Auto-start helper
# ---------------------------------------------------------------------------


def ensure_gateway_running(port: int = 8800, timeout: float = 30.0) -> str:
    """Ensure gateway is running, start if needed. Returns base URL.

    This is meant to be imported by PAOP tools:
        from model_gateway.server import ensure_gateway_running
        base_url = ensure_gateway_running()

    1. Try GET http://localhost:<port>/health
    2. If fails: start gateway as daemon subprocess
    3. Poll health every 500ms until timeout
    4. Return f"http://localhost:{port}/v1"
    """
    health_url = f"http://localhost:{port}/health"

    # 1. Check if already running
    try:
        with httpx.Client(timeout=2.0) as client:
            resp = client.get(health_url)
            if resp.status_code < 500:
                return f"http://localhost:{port}/v1"
    except Exception:
        pass

    # 2. Start gateway as daemon
    subprocess.Popen(
        ["gateway", "start", "--foreground", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    # 3. Poll until healthy or timeout
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(0.5)
        try:
            with httpx.Client(timeout=2.0) as client:
                resp = client.get(health_url)
                if resp.status_code < 500:
                    return f"http://localhost:{port}/v1"
        except Exception:
            pass

    raise TimeoutError(
        f"Gateway did not become healthy within {timeout}s on port {port}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(host: str = "127.0.0.1", port: int = 8800) -> None:
    uvicorn.run(app, host=host, port=port)
