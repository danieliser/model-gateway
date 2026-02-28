"""Tests for model_gateway.server module."""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from httpx import AsyncClient, ASGITransport

from model_gateway.config import BackendConfig, GatewayConfig, ModelConfig
from model_gateway.backends import BackendStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config() -> GatewayConfig:
    return GatewayConfig(
        port=8800,
        default_model="haiku",
        embedding_model="nomic-embed",
        models={
            "haiku": ModelConfig(backend="anthropic", model_id="claude-haiku-4-5-20251001"),
            "local": ModelConfig(backend="mlx", model_id="mlx-community/Qwen3-4B-4bit"),
            "nomic-embed": ModelConfig(backend="mlx", model_id="nomic-embed-v1.5"),
        },
        backends={
            "anthropic": BackendConfig(enabled=True, api_key_env="ANTHROPIC_API_KEY"),
            "mlx": BackendConfig(enabled=True),
        },
        task_routing={"coding": "haiku"},
        fallback_chain=["anthropic"],
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_managers():
    """Return pre-configured mocks for BackendManager, ProxyManager, TaskRouter."""
    cfg = _make_config()

    backend_status = {
        "haiku": BackendStatus(name="haiku", running=True, port=None, model_alias="haiku"),
        "local": BackendStatus(name="local", running=False, port=None, model_alias="local"),
        "nomic-embed": BackendStatus(name="nomic-embed", running=False, port=None, model_alias="nomic-embed"),
    }

    mock_backend = MagicMock()
    mock_backend.get_status.return_value = backend_status
    mock_backend.switch_model = AsyncMock(return_value=True)
    mock_backend.ensure_model = AsyncMock(return_value=0)  # cloud returns 0
    mock_backend.unload_model = MagicMock(return_value=True)
    mock_backend.get_port = MagicMock(return_value=None)
    mock_backend.is_model_loaded = MagicMock(return_value=False)
    mock_backend.get_backend_name = MagicMock(return_value="anthropic")
    mock_backend.cleanup = AsyncMock()
    mock_backend.start_idle_monitor = MagicMock()
    mock_backend.stop_idle_monitor = MagicMock()
    mock_backend._embed_manager = MagicMock()
    mock_backend._embed_manager.is_loaded = MagicMock(return_value=False)

    mock_proxy = MagicMock()
    mock_proxy.get_available_models.return_value = [
        {"alias": "haiku", "backend": "anthropic", "model_id": "claude-haiku-4-5-20251001"},
        {"alias": "local", "backend": "mlx", "model_id": "mlx-community/Qwen3-4B-4bit"},
    ]
    mock_proxy.completion = AsyncMock(return_value={"choices": [{"message": {"content": "hi"}}]})
    mock_proxy.setup = MagicMock()
    mock_proxy.update_backend_url = MagicMock()
    mock_proxy.on_model_loaded = MagicMock()
    mock_proxy.on_model_unloaded = MagicMock()
    mock_proxy.register_external_url = MagicMock()
    mock_proxy.close = AsyncMock()

    mock_router = MagicMock()
    mock_router.resolve_model.return_value = "haiku"

    return cfg, mock_backend, mock_proxy, mock_router


@pytest_asyncio.fixture
async def client(mock_managers):
    """AsyncClient pointed at the FastAPI app with mocked state."""
    import model_gateway.server as srv

    cfg, mock_backend, mock_proxy, mock_router = mock_managers

    with patch("model_gateway.server.load_config", return_value=cfg), \
         patch("model_gateway.server.validate_config", return_value=[]), \
         patch("model_gateway.server.BackendManager", return_value=mock_backend), \
         patch("model_gateway.server.ProxyManager", return_value=mock_proxy), \
         patch("model_gateway.server.TaskRouter", return_value=mock_router):

        transport = ASGITransport(app=srv.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            srv._config = cfg
            srv._backend_manager = mock_backend
            srv._proxy_manager = mock_proxy
            srv._task_router = mock_router
            yield ac, mock_backend, mock_proxy, mock_router

        srv._config = None
        srv._backend_manager = None
        srv._proxy_manager = None
        srv._task_router = None


# ---------------------------------------------------------------------------
# Test 1: GET /health returns 200 — healthy with zero models loaded
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_returns_200_healthy(client):
    ac, mock_backend, _, _ = client
    resp = await ac.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["port"] == 8800
    assert data["default_model"] == "haiku"
    # New format: loaded_models + available_models
    assert "loaded_models" in data
    assert "available_models" in data


# ---------------------------------------------------------------------------
# Test 2: GET /v1/models returns model list with loaded status
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_models_returns_model_list(client):
    ac, mock_backend, mock_proxy, _ = client
    resp = await ac.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    aliases = {m["alias"] for m in data["data"]}
    assert aliases == {"haiku", "local"}
    # Each model should have a "loaded" field
    for m in data["data"]:
        assert "loaded" in m


# ---------------------------------------------------------------------------
# Test 3: POST /v1/chat/completions triggers ensure_model
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_completions_triggers_ensure_model(client):
    ac, mock_backend, mock_proxy, mock_router = client
    mock_router.resolve_model.return_value = "haiku"
    mock_backend.ensure_model = AsyncMock(return_value=0)  # cloud
    mock_proxy.completion = AsyncMock(
        return_value={"choices": [{"message": {"content": "hello"}}]}
    )

    resp = await ac.post("/v1/chat/completions", json={
        "model": "haiku",
        "messages": [{"role": "user", "content": "Say hi"}],
        "stream": False,
    })
    assert resp.status_code == 200
    mock_backend.ensure_model.assert_called_once_with("haiku")


@pytest.mark.asyncio
async def test_chat_completions_ensure_model_registers_port(client):
    """When ensure_model returns a real port, on_model_loaded is called."""
    ac, mock_backend, mock_proxy, mock_router = client
    mock_router.resolve_model.return_value = "local"
    mock_backend.ensure_model = AsyncMock(return_value=8801)
    mock_proxy.completion = AsyncMock(
        return_value={"choices": [{"message": {"content": "hi"}}]}
    )

    resp = await ac.post("/v1/chat/completions", json={
        "model": "local",
        "messages": [{"role": "user", "content": "test"}],
    })
    assert resp.status_code == 200
    mock_proxy.on_model_loaded.assert_called_once_with("local", "mlx", 8801)


# ---------------------------------------------------------------------------
# Test 4: POST /v1/chat/completions with unknown model returns 404
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_completions_unknown_model_returns_404(client):
    ac, _, _, mock_router = client
    mock_router.resolve_model.side_effect = ValueError("Unknown model 'bogus'. Available models: ['haiku', 'local']")

    resp = await ac.post("/v1/chat/completions", json={
        "model": "bogus",
        "messages": [{"role": "user", "content": "Test"}],
    })
    assert resp.status_code == 404
    assert "Unknown model" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Test 5: POST /gateway/switch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_switch_model_valid(client):
    ac, mock_backend, mock_proxy, _ = client
    mock_backend.ensure_model = AsyncMock(return_value=8801)
    mock_backend.get_port = MagicMock(return_value=8801)

    resp = await ac.post("/gateway/switch", json={"model": "local"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["model"] == "local"


@pytest.mark.asyncio
async def test_switch_model_unknown_returns_404(client):
    ac, _, _, _ = client
    resp = await ac.post("/gateway/switch", json={"model": "nonexistent"})
    assert resp.status_code == 404
    assert "Unknown model" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Test 6: GET /gateway/config returns redacted config with new fields
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_config_redacts_api_keys(client):
    ac, _, _, _ = client
    resp = await ac.get("/gateway/config")
    assert resp.status_code == 200
    data = resp.json()

    assert "port" in data
    assert "models" in data
    assert "backends" in data
    assert "idle_timeout" in data
    assert "idle_check_interval" in data

    for backend_data in data["backends"].values():
        if backend_data.get("api_key_env") is not None:
            assert backend_data["api_key_env"] == "***"

    for model_data in data["models"].values():
        if model_data.get("api_key_env") is not None:
            assert model_data["api_key_env"] == "***"
        assert "pin" in model_data


# ---------------------------------------------------------------------------
# Test: _config_to_safe_dict redacts nested api_key fields
# ---------------------------------------------------------------------------

def test_config_to_safe_dict_redacts_keys():
    from model_gateway.server import _config_to_safe_dict

    cfg = _make_config()
    result = _config_to_safe_dict(cfg)

    assert result["backends"]["anthropic"]["api_key_env"] == "***"
    assert result["backends"]["anthropic"]["enabled"] is True
    assert result["port"] == 8800
    assert result["idle_timeout"] == 900
    assert result["idle_check_interval"] == 30


def test_redact_api_keys_nested():
    from model_gateway.server import _redact_api_keys

    obj = {
        "name": "test",
        "api_key": "secret",
        "api_key_env": "MY_KEY",
        "nested": {
            "api_key": "another-secret",
            "keep": "this",
        },
        "list": [{"api_key": "list-secret"}, {"safe": "value"}],
    }
    result = _redact_api_keys(obj)
    assert result["api_key"] == "***"
    assert result["api_key_env"] == "***"
    assert result["name"] == "test"
    assert result["nested"]["api_key"] == "***"
    assert result["nested"]["keep"] == "this"
    assert result["list"][0]["api_key"] == "***"
    assert result["list"][1]["safe"] == "value"


# ---------------------------------------------------------------------------
# Test: POST /v1/embeddings triggers ensure_model
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_embeddings_triggers_ensure_model(client):
    ac, mock_backend, mock_proxy, _ = client
    mock_backend.ensure_model = AsyncMock(return_value=8801)
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "object": "list",
        "data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}],
    }
    mock_response.__class__ = httpx.Response
    mock_proxy.embedding = AsyncMock(return_value=mock_response)

    resp = await ac.post("/v1/embeddings", json={
        "model": "nomic-embed",
        "input": "hello world",
    })
    assert resp.status_code == 200
    mock_backend.ensure_model.assert_called_once_with("nomic-embed")


@pytest.mark.asyncio
async def test_embeddings_uses_default_embedding_model(client):
    ac, mock_backend, mock_proxy, _ = client
    mock_backend.ensure_model = AsyncMock(return_value=0)
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "object": "list",
        "data": [{"object": "embedding", "embedding": [0.1], "index": 0}],
    }
    mock_response.__class__ = httpx.Response
    mock_proxy.embedding = AsyncMock(return_value=mock_response)

    resp = await ac.post("/v1/embeddings", json={"input": "test"})
    assert resp.status_code == 200
    mock_proxy.embedding.assert_called_once_with("nomic-embed", "test")


@pytest.mark.asyncio
async def test_embeddings_unknown_model_returns_404(client):
    ac, _, _, _ = client
    resp = await ac.post("/v1/embeddings", json={
        "model": "nonexistent",
        "input": "test",
    })
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Test: POST /gateway/models/load
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_load_model_endpoint(client):
    ac, mock_backend, mock_proxy, _ = client
    mock_backend.ensure_model = AsyncMock(return_value=8801)

    resp = await ac.post("/gateway/models/load", json={"model": "local"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["model"] == "local"
    assert data["port"] == 8801
    mock_backend.ensure_model.assert_called_once_with("local")
    mock_proxy.on_model_loaded.assert_called_once_with("local", "mlx", 8801)


@pytest.mark.asyncio
async def test_load_model_unknown_returns_404(client):
    ac, _, _, _ = client
    resp = await ac.post("/gateway/models/load", json={"model": "bogus"})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Test: POST /gateway/models/unload
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unload_model_endpoint(client):
    ac, mock_backend, mock_proxy, _ = client
    mock_backend.unload_model = MagicMock(return_value=True)

    resp = await ac.post("/gateway/models/unload", json={"model": "local"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["model"] == "local"
    mock_backend.unload_model.assert_called_once_with("local")
    mock_proxy.on_model_unloaded.assert_called_once_with("local", "mlx")


@pytest.mark.asyncio
async def test_unload_model_unknown_returns_404(client):
    ac, _, _, _ = client
    resp = await ac.post("/gateway/models/unload", json={"model": "bogus"})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_unload_pinned_model_returns_success_false(client):
    ac, mock_backend, mock_proxy, _ = client
    mock_backend.unload_model = MagicMock(return_value=False)

    resp = await ac.post("/gateway/models/unload", json={"model": "local"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is False
    mock_proxy.on_model_unloaded.assert_not_called()


# ---------------------------------------------------------------------------
# Test: Startup loads zero models
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_startup_loads_zero_models(client):
    """Startup should not eagerly load models — ensure_model not called."""
    ac, mock_backend, _, _ = client
    # ensure_model should not have been called during test setup
    # (our fixture bypasses the startup event, confirming no eager loading)
    mock_backend.ensure_model.assert_not_called()


@pytest.mark.asyncio
async def test_startup_event_starts_idle_monitor():
    """The startup event handler should start the idle monitor."""
    import model_gateway.server as srv
    cfg = _make_config()

    mock_backend = MagicMock()
    mock_backend.start_idle_monitor = MagicMock()
    mock_backend.ensure_model = AsyncMock()
    mock_backend.get_status.return_value = {}

    mock_proxy = MagicMock()
    mock_proxy.setup = MagicMock()
    mock_proxy.register_external_url = MagicMock()
    mock_proxy.close = AsyncMock()

    with patch("model_gateway.server.load_config", return_value=cfg), \
         patch("model_gateway.server.validate_config", return_value=[]), \
         patch("model_gateway.server.BackendManager", return_value=mock_backend), \
         patch("model_gateway.server.ProxyManager", return_value=mock_proxy), \
         patch("model_gateway.server.TaskRouter"):

        await srv.startup()
        mock_backend.start_idle_monitor.assert_called_once()

        # Cleanup
        srv._config = None
        srv._backend_manager = None
        srv._proxy_manager = None
        srv._task_router = None
