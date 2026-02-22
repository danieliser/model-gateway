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
        models={
            "haiku": ModelConfig(backend="anthropic", model_id="claude-haiku-4-5-20251001"),
            "local": ModelConfig(backend="mlx", model_id="mlx-community/Qwen3-4B-4bit"),
        },
        backends={
            "anthropic": BackendConfig(enabled=True, api_key_env="ANTHROPIC_API_KEY"),
            "mlx": BackendConfig(enabled=True),
        },
        task_routing={"coding": "haiku"},
        fallback_chain=["anthropic"],
    )


def _make_backend_status(running: bool = False, port: int | None = None, model_alias: str | None = None) -> BackendStatus:
    return BackendStatus(name="anthropic", running=running, port=port, model_alias=model_alias)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_managers():
    """Return pre-configured mocks for BackendManager, ProxyManager, TaskRouter."""
    cfg = _make_config()

    backend_status = {
        "anthropic": BackendStatus(name="anthropic", running=True, port=None, model_alias="haiku"),
        "mlx": BackendStatus(name="mlx", running=False, port=None),
    }

    mock_backend = MagicMock()
    mock_backend.get_status.return_value = backend_status
    mock_backend.switch_model = AsyncMock(return_value=True)
    mock_backend.cleanup = AsyncMock()

    mock_proxy = MagicMock()
    mock_proxy.get_available_models.return_value = [
        {"alias": "haiku", "backend": "anthropic", "model_id": "claude-haiku-4-5-20251001"},
        {"alias": "local", "backend": "mlx", "model_id": "mlx-community/Qwen3-4B-4bit"},
    ]
    mock_proxy.completion = AsyncMock(return_value={"choices": [{"message": {"content": "hi"}}]})
    mock_proxy.setup = MagicMock()
    mock_proxy.update_backend_url = MagicMock()

    mock_router = MagicMock()
    mock_router.resolve_model.return_value = "haiku"

    return cfg, mock_backend, mock_proxy, mock_router


@pytest_asyncio.fixture
async def client(mock_managers):
    """AsyncClient pointed at the FastAPI app with mocked state."""
    import model_gateway.server as srv

    cfg, mock_backend, mock_proxy, mock_router = mock_managers

    # Patch load_config + validate_config so startup doesn't touch filesystem or litellm
    with patch("model_gateway.server.load_config", return_value=cfg), \
         patch("model_gateway.server.validate_config", return_value=[]), \
         patch("model_gateway.server.BackendManager", return_value=mock_backend), \
         patch("model_gateway.server.ProxyManager", return_value=mock_proxy), \
         patch("model_gateway.server.TaskRouter", return_value=mock_router):

        transport = ASGITransport(app=srv.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            # Manually set global state (bypass lifespan for simplicity)
            srv._config = cfg
            srv._backend_manager = mock_backend
            srv._proxy_manager = mock_proxy
            srv._task_router = mock_router
            yield ac, mock_backend, mock_proxy, mock_router

        # Reset global state after each test
        srv._config = None
        srv._backend_manager = None
        srv._proxy_manager = None
        srv._task_router = None


# ---------------------------------------------------------------------------
# Test 1: GET /health returns 200 with backend status
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_returns_200_with_backends(client):
    ac, mock_backend, _, _ = client
    resp = await ac.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["port"] == 8800
    assert data["default_model"] == "haiku"
    assert "anthropic" in data["backends"]
    assert "mlx" in data["backends"]
    assert data["backends"]["anthropic"]["running"] is True
    assert data["backends"]["mlx"]["running"] is False


# ---------------------------------------------------------------------------
# Test 2: GET /v1/models returns model list
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_models_returns_model_list(client):
    ac, _, mock_proxy, _ = client
    resp = await ac.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    aliases = {m["alias"] for m in data["data"]}
    assert aliases == {"haiku", "local"}


# ---------------------------------------------------------------------------
# Test 3: POST /v1/chat/completions with known model (mock proxy completion)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_completions_non_streaming(client):
    ac, _, mock_proxy, mock_router = client
    mock_router.resolve_model.return_value = "haiku"
    mock_proxy.completion = AsyncMock(
        return_value={"choices": [{"message": {"content": "hello"}}]}
    )

    resp = await ac.post("/v1/chat/completions", json={
        "model": "haiku",
        "messages": [{"role": "user", "content": "Say hi"}],
        "stream": False,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "choices" in data
    mock_proxy.completion.assert_called_once_with("haiku", [{"role": "user", "content": "Say hi"}])


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
# Test 5: POST /gateway/switch with valid model
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_switch_model_valid(client):
    ac, mock_backend, mock_proxy, _ = client
    # After switch, backend reports a port
    mock_backend.get_status.return_value = {
        "anthropic": BackendStatus(name="anthropic", running=True, port=8801, model_alias="haiku"),
        "mlx": BackendStatus(name="mlx", running=False),
    }
    mock_backend.switch_model = AsyncMock(return_value=True)

    resp = await ac.post("/gateway/switch", json={"model": "haiku"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["model"] == "haiku"
    mock_backend.switch_model.assert_called_once_with("anthropic", "haiku")


@pytest.mark.asyncio
async def test_switch_model_unknown_returns_404(client):
    ac, _, _, _ = client
    resp = await ac.post("/gateway/switch", json={"model": "nonexistent"})
    assert resp.status_code == 404
    assert "Unknown model" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Test 6: GET /gateway/config returns redacted config
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_config_redacts_api_keys(client):
    ac, _, _, _ = client
    resp = await ac.get("/gateway/config")
    assert resp.status_code == 200
    data = resp.json()

    # Check structure
    assert "port" in data
    assert "models" in data
    assert "backends" in data

    # api_key_env values should be redacted
    for backend_data in data["backends"].values():
        if backend_data.get("api_key_env") is not None:
            assert backend_data["api_key_env"] == "***"

    for model_data in data["models"].values():
        if model_data.get("api_key_env") is not None:
            assert model_data["api_key_env"] == "***"


# ---------------------------------------------------------------------------
# Test: _config_to_safe_dict redacts nested api_key fields
# ---------------------------------------------------------------------------

def test_config_to_safe_dict_redacts_keys():
    from model_gateway.server import _config_to_safe_dict

    cfg = _make_config()
    result = _config_to_safe_dict(cfg)

    # Backends with api_key_env should have it redacted
    assert result["backends"]["anthropic"]["api_key_env"] == "***"
    # Non-api-key fields should be preserved
    assert result["backends"]["anthropic"]["enabled"] is True
    assert result["port"] == 8800


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
# Test: _get_backend_ports only includes ports that are not None
# ---------------------------------------------------------------------------

def test_get_backend_ports_filters_none():
    import model_gateway.server as srv
    from model_gateway.backends import BackendStatus

    mock_backend = MagicMock()
    mock_backend.get_status.return_value = {
        "mlx": BackendStatus(name="mlx", running=True, port=8801),
        "anthropic": BackendStatus(name="anthropic", running=True, port=None),
    }
    srv._backend_manager = mock_backend

    ports = srv._get_backend_ports()
    assert ports == {"mlx": 8801}
    assert "anthropic" not in ports

    srv._backend_manager = None
