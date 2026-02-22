"""Tests for model_gateway.proxy module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

from model_gateway.config import BackendConfig, GatewayConfig, ModelConfig
from model_gateway.proxy import ProxyManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_config() -> GatewayConfig:
    return GatewayConfig(
        port=8800,
        default_model="local",
        models={
            "local": ModelConfig(backend="mlx", model_id="mlx-community/Qwen3-4B-4bit"),
            "haiku": ModelConfig(backend="anthropic", model_id="claude-haiku-4-5-20251001"),
            "gpt4o": ModelConfig(backend="openai", model_id="gpt-4o"),
        },
        backends={
            "mlx": BackendConfig(enabled=True),
            "anthropic": BackendConfig(enabled=True, api_key_env="ANTHROPIC_API_KEY"),
            "openai": BackendConfig(enabled=True, api_key_env="OPENAI_API_KEY"),
        },
        task_routing={"coding": "haiku"},
        fallback_chain=["anthropic", "mlx"],
    )


def _get_model_list(proxy: ProxyManager) -> list[dict]:
    """Extract the model_list that would be passed to the Router."""
    config = proxy._config
    entries = []
    for alias, model_cfg in config.models.items():
        entry = proxy._build_model_entry(alias, model_cfg)
        if entry is not None:
            entries.append(entry)
    return entries


# ---------------------------------------------------------------------------
# Test 1: _build_model_entry for local backends
# ---------------------------------------------------------------------------

def test_build_entry_local_backend():
    config = _make_config()
    proxy = ProxyManager(config)
    proxy._backend_urls["mlx"] = "http://localhost:8801/v1"

    entry = proxy._build_model_entry("local", config.models["local"])
    assert entry is not None
    params = entry["litellm_params"]
    assert params["model"] == "openai/mlx-community/Qwen3-4B-4bit"
    assert params["api_base"] == "http://localhost:8801/v1"
    assert params["api_key"] == "not-needed"


# ---------------------------------------------------------------------------
# Test 2: _build_model_entry for Anthropic
# ---------------------------------------------------------------------------

def test_build_entry_anthropic_backend():
    config = _make_config()
    proxy = ProxyManager(config)

    entry = proxy._build_model_entry("haiku", config.models["haiku"])
    assert entry is not None
    params = entry["litellm_params"]
    assert params["model"] == "anthropic/claude-haiku-4-5-20251001"
    assert "ANTHROPIC_API_KEY" in params["api_key"]
    assert "api_base" not in params


# ---------------------------------------------------------------------------
# Test 3: _build_model_entry for OpenAI
# ---------------------------------------------------------------------------

def test_build_entry_openai_backend():
    config = _make_config()
    proxy = ProxyManager(config)

    entry = proxy._build_model_entry("gpt4o", config.models["gpt4o"])
    assert entry is not None
    params = entry["litellm_params"]
    assert params["model"] == "gpt-4o"
    assert "OPENAI_API_KEY" in params["api_key"]
    assert "api_base" not in params


# ---------------------------------------------------------------------------
# Test 4: setup() creates Router with model_list
# ---------------------------------------------------------------------------

def test_setup_creates_router():
    with patch("model_gateway.proxy.Router") as MockRouter:
        config = _make_config()
        proxy = ProxyManager(config)
        proxy.setup({"mlx": 8801})

        MockRouter.assert_called_once()
        model_list = MockRouter.call_args[1]["model_list"]
        aliases = [e["model_name"] for e in model_list]
        assert "local" in aliases
        assert "haiku" in aliases
        assert "gpt4o" in aliases


# ---------------------------------------------------------------------------
# Test 5: local model with no port is omitted
# ---------------------------------------------------------------------------

def test_local_model_without_port_omitted():
    with patch("model_gateway.proxy.Router") as MockRouter:
        config = _make_config()
        proxy = ProxyManager(config)
        proxy.setup({})  # no ports

        model_list = MockRouter.call_args[1]["model_list"]
        aliases = [e["model_name"] for e in model_list]
        assert "local" not in aliases
        assert "haiku" in aliases
        assert "gpt4o" in aliases


# ---------------------------------------------------------------------------
# Test 6: get_available_models returns all configured models
# ---------------------------------------------------------------------------

def test_get_available_models_returns_all():
    config = _make_config()
    proxy = ProxyManager(config)
    proxy._backend_urls["mlx"] = "http://localhost:8801/v1"

    models = proxy.get_available_models()
    aliases = {m["alias"] for m in models}
    assert aliases == {"local", "haiku", "gpt4o"}


def test_get_available_models_includes_backend_info():
    config = _make_config()
    proxy = ProxyManager(config)
    proxy._backend_urls["mlx"] = "http://localhost:8801/v1"

    models = proxy.get_available_models()
    local_info = next(m for m in models if m["alias"] == "local")
    assert local_info["backend"] == "mlx"
    assert local_info["model_id"] == "mlx-community/Qwen3-4B-4bit"
    assert local_info["api_base"] == "http://localhost:8801/v1"

    haiku_info = next(m for m in models if m["alias"] == "haiku")
    assert haiku_info["backend"] == "anthropic"
    assert "api_base" not in haiku_info


# ---------------------------------------------------------------------------
# Test 7: completion() calls router.acompletion
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_completion_calls_router_acompletion():
    with patch("model_gateway.proxy.Router") as MockRouter:
        mock_router = MagicMock()
        fake_response = MagicMock()
        mock_router.acompletion = AsyncMock(return_value=fake_response)
        MockRouter.return_value = mock_router

        config = _make_config()
        proxy = ProxyManager(config)
        proxy.setup({"mlx": 8801})

        messages = [{"role": "user", "content": "Hello"}]
        result = await proxy.completion("haiku", messages, stream=False)

        mock_router.acompletion.assert_called_once_with(
            model="haiku",
            messages=messages,
            stream=False,
        )
        assert result is fake_response


@pytest.mark.asyncio
async def test_completion_cloud_passes_extra_kwargs():
    with patch("model_gateway.proxy.Router") as MockRouter:
        mock_router = MagicMock()
        fake_response = MagicMock()
        mock_router.acompletion = AsyncMock(return_value=fake_response)
        MockRouter.return_value = mock_router

        config = _make_config()
        proxy = ProxyManager(config)
        proxy.setup({"mlx": 8801})

        messages = [{"role": "user", "content": "Summarize this"}]
        result = await proxy.completion("haiku", messages, stream=True, temperature=0.7)

        mock_router.acompletion.assert_called_once_with(
            model="haiku",
            messages=messages,
            stream=True,
            temperature=0.7,
        )
        assert result is fake_response


@pytest.mark.asyncio
async def test_completion_raises_without_setup():
    config = _make_config()
    proxy = ProxyManager(config)

    # Cloud model with no router set up should raise
    with pytest.raises(RuntimeError, match="not initialized"):
        await proxy.completion("haiku", [{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------------------
# Test 8: Local pass-through bypasses LiteLLM
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_local_passthrough_bypasses_litellm():
    """Local backend completion should use httpx directly, not LiteLLM Router."""
    with patch("model_gateway.proxy.Router") as MockRouter:
        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock()
        MockRouter.return_value = mock_router

        config = _make_config()
        proxy = ProxyManager(config)
        proxy.setup({"mlx": 8801})

        # Mock the httpx client
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "hi"}}]}
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        proxy._http_client = mock_http

        messages = [{"role": "user", "content": "Hello"}]
        result = await proxy.completion("local", messages, stream=False)

        # LiteLLM should NOT be called
        mock_router.acompletion.assert_not_called()
        # httpx should be called with the backend URL
        mock_http.post.assert_called_once()
        call_url = mock_http.post.call_args[0][0]
        assert "localhost:8801" in call_url
        assert result is mock_response


@pytest.mark.asyncio
async def test_is_local_model():
    config = _make_config()
    proxy = ProxyManager(config)
    proxy._backend_urls["mlx"] = "http://localhost:8801/v1"

    is_local, api_base = proxy._is_local_model("local")
    assert is_local is True
    assert api_base == "http://localhost:8801/v1"

    is_local, api_base = proxy._is_local_model("haiku")
    assert is_local is False

    is_local, api_base = proxy._is_local_model("nonexistent")
    assert is_local is False
