"""FastAPI/uvicorn server setup and lifecycle management for the gateway."""

import json
import logging
import subprocess
import time
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from model_gateway.backends import BackendManager
from model_gateway.config import CLOUD_BACKENDS, TTS_BACKENDS, GatewayConfig, load_config, validate_config
from model_gateway.proxy import ProxyManager
from model_gateway.routing import TaskRouter

logger = logging.getLogger(__name__)

app = FastAPI(title="Model Gateway", version="0.2.0")

# Global state (initialized on startup)
_config: GatewayConfig | None = None
_backend_manager: BackendManager | None = None
_proxy_manager: ProxyManager | None = None
_task_router: TaskRouter | None = None


@app.on_event("startup")
async def startup() -> None:
    """Load config, init managers. No models loaded — everything is lazy."""
    global _config, _backend_manager, _proxy_manager, _task_router

    _config = load_config()
    errors = validate_config(_config)
    if any(e.startswith("ERROR") for e in errors):
        raise RuntimeError(f"Config errors: {errors}")

    _backend_manager = BackendManager(_config)
    _proxy_manager = ProxyManager(_config, ensure_model_fn=_backend_manager.ensure_model)
    _task_router = TaskRouter(_config)

    # Register external backend URLs (lm-studio, ollama, etc.)
    external_urls = _get_external_backend_urls()
    for backend_name, url in external_urls.items():
        _proxy_manager.register_external_url(backend_name, url)

    # Setup proxy for cloud backends (no local ports yet — lazy loading)
    _proxy_manager.setup({})

    # Configure TTS subsystems
    from model_gateway.normalize import configure as configure_normalize
    from model_gateway.tts_cache import configure as configure_cache
    configure_normalize(_config.tts.pronunciations)
    configure_cache(
        cache_dir=_config.tts.cache_dir,
        max_mb=_config.tts.cache_max_mb,
        max_age_days=_config.tts.cache_max_age_days,
    )

    # Start idle monitor
    _backend_manager.start_idle_monitor()


@app.on_event("shutdown")
async def shutdown() -> None:
    if _backend_manager:
        _backend_manager.stop_idle_monitor()
    if _proxy_manager:
        await _proxy_manager.close()
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

    # Lazy-load: ensure model is running before routing
    port = await _backend_manager.ensure_model(resolved_model)
    if port is not None and port > 0:
        model_cfg = _config.models.get(resolved_model)
        if model_cfg:
            _proxy_manager.on_model_loaded(resolved_model, model_cfg.backend, port)

    # In-process MLX chat models — call directly, no HTTP proxy
    if _backend_manager._llm_manager.is_loaded(resolved_model):
        body_kwargs = {
            k: body[k]
            for k in ("max_tokens", "temperature", "top_p", "repetition_penalty", "stop")
            if k in body
        }
        if stream:
            async def stream_inprocess():
                async for chunk in _backend_manager._llm_manager.stream_generate(
                    resolved_model, messages, **body_kwargs
                ):
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(stream_inprocess(), media_type="text/event-stream")
        else:
            result = await _backend_manager._llm_manager.generate(
                resolved_model, messages, **body_kwargs
            )
            return JSONResponse(content=result)

    # Call through proxy (cloud, external, llama-cpp)
    result = await _proxy_manager.completion(resolved_model, messages, stream=stream)

    # httpx.Response = local pass-through, otherwise LiteLLM ModelResponse
    if isinstance(result, httpx.Response):
        if stream:
            async def stream_raw():
                async for line in result.aiter_lines():
                    yield f"{line}\n"
                await result.aclose()

            return StreamingResponse(stream_raw(), media_type="text/event-stream")
        else:
            return JSONResponse(content=result.json())
    else:
        if stream:
            async def generate():
                async for chunk in result:
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            content = result.model_dump() if hasattr(result, "model_dump") else result
            return JSONResponse(content=content)


@app.post("/v1/embeddings", response_model=None)
async def embeddings(request: Request) -> JSONResponse:
    if not _config or not _proxy_manager or not _backend_manager:
        raise HTTPException(503, "Gateway not initialized yet")

    body = await request.json()
    model = body.get("model", _config.embedding_model or _config.default_model)
    input_text = body.get("input", "")

    if model not in _config.models:
        raise HTTPException(404, f"Unknown model: {model}")

    # Lazy-load: ensure model is running
    port = await _backend_manager.ensure_model(model)
    if port is not None and port > 0:
        model_cfg = _config.models.get(model)
        if model_cfg:
            _proxy_manager.on_model_loaded(model, model_cfg.backend, port)

    # In-process embedding models — call directly, no HTTP
    model_cfg = _config.models.get(model)
    if model_cfg and _backend_manager._embed_manager.is_loaded(model):
        result = _backend_manager._embed_manager.generate(model, input_text)
        return JSONResponse(content=result)

    result = await _proxy_manager.embedding(model, input_text)

    if isinstance(result, httpx.Response):
        return JSONResponse(content=result.json())
    else:
        content = result.model_dump() if hasattr(result, "model_dump") else result
        return JSONResponse(content=content)


@app.post("/v1/audio/speech", response_model=None)
async def audio_speech(request: Request) -> Response:
    """OpenAI-compatible TTS endpoint. Returns audio bytes."""
    if not _config or not _backend_manager or not _proxy_manager:
        raise HTTPException(503, "Gateway not initialized yet")

    body = await request.json()
    text = body.get("input", "")
    if not text:
        raise HTTPException(400, "Missing 'input' field")

    model = body.get("model")
    voice = body.get("voice", _config.tts.default_voice)
    speed = float(body.get("speed", _config.tts.default_speed))
    lang_code = body.get("lang_code", _config.tts.default_lang_code)
    exaggeration = body.get("exaggeration")
    instruct = body.get("instruct")
    conds_path = body.get("conds_path")
    ref_audio = body.get("ref_audio")
    response_format = body.get("response_format", "wav")

    if exaggeration is not None:
        exaggeration = float(exaggeration)

    # Resolve model — accept config alias, model_id, or filesystem path
    if not model:
        for alias, cfg in _config.models.items():
            if cfg.backend in TTS_BACKENDS:
                model = alias
                break
    elif model not in _config.models:
        # Try reverse lookup by model_id (e.g. "kokoro-82m-bf16" → "kokoro")
        for alias, cfg in _config.models.items():
            if cfg.backend in TTS_BACKENDS and cfg.model_id == model:
                model = alias
                break
        # Try matching by basename of a filesystem path
        if model not in _config.models and "/" in model:
            basename = model.rstrip("/").rsplit("/", 1)[-1]
            for alias, cfg in _config.models.items():
                if cfg.backend in TTS_BACKENDS and cfg.model_id == basename:
                    model = alias
                    break
    if not model or model not in _config.models:
        raise HTTPException(404, f"Unknown or no TTS model: {model}")

    model_cfg = _config.models[model]
    if model_cfg.backend not in TTS_BACKENDS:
        raise HTTPException(400, f"Model '{model}' is not a TTS model (backend: {model_cfg.backend})")

    # Check cache first
    from model_gateway.tts_cache import cache_lookup, cache_store
    model_id = model_cfg.model_id or model
    cached = cache_lookup(text, model_id, voice, lang_code, speed, conds_path)
    if cached:
        audio_bytes = cached.read_bytes()
        media = "audio/mpeg" if response_format == "mp3" else "audio/wav"
        return Response(content=audio_bytes, media_type=media)

    # Lazy-load model
    port = await _backend_manager.ensure_model(model)
    if port is None:
        raise HTTPException(503, f"Failed to load TTS model: {model}")

    # In-process MLX-Audio — call TtsManager directly
    if model_cfg.backend == "mlx-audio" and _backend_manager._tts_manager.is_loaded(model):
        # Apply text normalization
        from model_gateway.normalize import normalize
        normalized = normalize(text)

        audio_bytes = _backend_manager._tts_manager.generate(
            model,
            normalized,
            voice=voice,
            speed=speed,
            lang_code=lang_code,
            exaggeration=exaggeration,
            instruct=instruct,
            conds_path=conds_path,
            ref_audio=ref_audio,
            response_format=response_format,
        )

        # Store in cache
        cache_store(text, model_id, voice, lang_code, speed, audio_bytes, conds_path=conds_path)

        media = "audio/mpeg" if response_format == "mp3" else "audio/wav"
        return Response(content=audio_bytes, media_type=media)

    # Cloud or external TTS — proxy through
    audio_bytes = await _proxy_manager.audio_speech(
        model=model,
        text=text,
        voice=voice,
        speed=speed,
        response_format=response_format,
        lang_code=lang_code,
        exaggeration=exaggeration,
        instruct=instruct,
        conds_path=conds_path,
        ref_audio=ref_audio,
    )

    # Store in cache
    cache_store(text, model_id, voice, lang_code, speed, audio_bytes, conds_path=conds_path)

    media = "audio/mpeg" if response_format == "mp3" else "audio/wav"
    return Response(content=audio_bytes, media_type=media)


@app.get("/v1/audio/cache/stats")
async def audio_cache_stats() -> dict:
    """Return TTS cache statistics."""
    from model_gateway.tts_cache import cache_stats
    return cache_stats()


@app.post("/v1/audio/cache/prune")
async def audio_cache_prune() -> dict:
    """Run TTS cache eviction."""
    from model_gateway.tts_cache import cache_prune
    return cache_prune()


@app.get("/v1/models")
async def list_models() -> dict:
    models = _proxy_manager.get_available_models()
    # Annotate with loaded status
    for m in models:
        alias = m["alias"]
        if _backend_manager:
            m["loaded"] = _backend_manager.is_model_loaded(alias)
        else:
            m["loaded"] = False
    return {"object": "list", "data": models}


@app.get("/health")
async def health() -> dict:
    statuses = _backend_manager.get_status()
    loaded = {
        alias: {
            "running": s.running,
            "port": s.port,
            "model": s.model_alias,
            "last_used": s.last_used,
        }
        for alias, s in statuses.items()
        if s.running and (s.port is not None or s.last_used is not None)
    }
    available = [
        alias for alias, s in statuses.items()
        if alias not in loaded
    ]
    return {
        "status": "healthy",
        "port": _config.port,
        "default_model": _config.default_model,
        "loaded_models": loaded,
        "available_models": available,
    }


@app.get("/gateway/config")
async def get_config() -> dict:
    return _config_to_safe_dict(_config)


@app.post("/gateway/switch")
async def switch_model(request: Request) -> dict:
    body = await request.json()
    new_model = body.get("model")
    old_model = body.get("old_model")

    if new_model not in _config.models:
        raise HTTPException(404, f"Unknown model: {new_model}")

    if old_model and old_model in _config.models:
        success = await _backend_manager.switch_model(old_model, new_model)
    else:
        port = await _backend_manager.ensure_model(new_model)
        success = port is not None

    if success:
        port = _backend_manager.get_port(new_model)
        model_cfg = _config.models[new_model]
        if port and port > 0:
            _proxy_manager.on_model_loaded(new_model, model_cfg.backend, port)

    return {"success": success, "model": new_model}


@app.post("/gateway/models/load")
async def load_model(request: Request) -> dict:
    """Explicitly trigger model loading."""
    if not _config or not _backend_manager or not _proxy_manager:
        raise HTTPException(503, "Gateway not initialized yet")

    body = await request.json()
    model_alias = body.get("model")
    if not model_alias or model_alias not in _config.models:
        raise HTTPException(404, f"Unknown model: {model_alias}")

    port = await _backend_manager.ensure_model(model_alias)
    if port is not None and port > 0:
        model_cfg = _config.models[model_alias]
        _proxy_manager.on_model_loaded(model_alias, model_cfg.backend, port)

    return {
        "success": port is not None,
        "model": model_alias,
        "port": port,
    }


@app.post("/gateway/models/unload")
async def unload_model(request: Request) -> dict:
    """Explicitly unload a model."""
    if not _config or not _backend_manager or not _proxy_manager:
        raise HTTPException(503, "Gateway not initialized yet")

    body = await request.json()
    model_alias = body.get("model")
    if not model_alias or model_alias not in _config.models:
        raise HTTPException(404, f"Unknown model: {model_alias}")

    success = await _backend_manager.unload_model_async(model_alias)
    if success:
        model_cfg = _config.models[model_alias]
        _proxy_manager.on_model_unloaded(model_alias, model_cfg.backend)

    return {"success": success, "model": model_alias}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_external_backend_urls() -> dict[str, str]:
    """Get URLs for external backends (lm-studio, ollama, etc.) from config."""
    from model_gateway.backends import _EXTERNAL_BACKENDS
    urls: dict[str, str] = {}
    for name, cfg in _config.backends.items():
        if name in _EXTERNAL_BACKENDS and cfg.enabled and cfg.host:
            urls[name] = f"{cfg.host.rstrip('/')}/v1"
    return urls


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
        "embedding_model": config.embedding_model,
        "idle_timeout": config.idle_timeout,
        "idle_check_interval": config.idle_check_interval,
        "models": {
            name: {
                "backend": m.backend,
                "model_id": m.model_id,
                "model_path": m.model_path,
                "api_key_env": m.api_key_env,
                "pin": m.pin,
                "idle_timeout": m.idle_timeout,
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
        "tts": {
            "pronunciations": config.tts.pronunciations,
            "cache_dir": config.tts.cache_dir,
            "cache_max_mb": config.tts.cache_max_mb,
            "default_voice": config.tts.default_voice,
            "default_lang_code": config.tts.default_lang_code,
            "default_speed": config.tts.default_speed,
        },
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
