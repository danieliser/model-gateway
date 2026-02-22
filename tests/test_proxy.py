"""Tests for model_gateway.proxy module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

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


# ---------------------------------------------------------------------------
# Test 1: setup() generates correct litellm model_list for local backends
# ---------------------------------------------------------------------------

def test_setup_local_backend_model_list():
    with patch("model_gateway.proxy.litellm") as mock_litellm:
        mock_litellm.model_list = []
        config = _make_config()
        proxy = ProxyManager(config)
        proxy.setup({"mlx": 8801})

        model_list = mock_litellm.model_list
        local_entry = next((e for e in model_list if e["model_name"] == "local"), None)
        assert local_entry is not None

        params = local_entry["litellm_params"]
        assert params["model"] == "openai/mlx-community/Qwen3-4B-4bit"
        assert params["api_base"] == "http://localhost:8801/v1"
        assert params["api_key"] == "not-needed"


# ---------------------------------------------------------------------------
# Test 2: setup() generates correct model_list for Anthropic
# ---------------------------------------------------------------------------

def test_setup_anthropic_backend_model_list():
    with patch("model_gateway.proxy.litellm") as mock_litellm:
        mock_litellm.model_list = []
        config = _make_config()
        proxy = ProxyManager(config)
        proxy.setup({"mlx": 8801})

        model_list = mock_litellm.model_list
        haiku_entry = next((e for e in model_list if e["model_name"] == "haiku"), None)
        assert haiku_entry is not None

        params = haiku_entry["litellm_params"]
        assert params["model"] == "anthropic/claude-haiku-4-5-20251001"
        assert "api_key" in params
        assert "ANTHROPIC_API_KEY" in params["api_key"]
        # Should NOT have api_base
        assert "api_base" not in params


# ---------------------------------------------------------------------------
# Test 3: setup() generates correct model_list for OpenAI
# ---------------------------------------------------------------------------

def test_setup_openai_backend_model_list():
    with patch("model_gateway.proxy.litellm") as mock_litellm:
        mock_litellm.model_list = []
        config = _make_config()
        proxy = ProxyManager(config)
        proxy.setup({"mlx": 8801})

        model_list = mock_litellm.model_list
        gpt4o_entry = next((e for e in model_list if e["model_name"] == "gpt4o"), None)
        assert gpt4o_entry is not None

        params = gpt4o_entry["litellm_params"]
        assert params["model"] == "gpt-4o"
        assert "OPENAI_API_KEY" in params["api_key"]
        # Should NOT have api_base
        assert "api_base" not in params


# ---------------------------------------------------------------------------
# Test 4: update_backend_url changes the URL for a backend
# ---------------------------------------------------------------------------

def test_update_backend_url_changes_api_base():
    with patch("model_gateway.proxy.litellm") as mock_litellm:
        mock_litellm.model_list = []
        config = _make_config()
        proxy = ProxyManager(config)
        proxy.setup({"mlx": 8801})

        # Grab the model_list and verify initial URL
        local_entry = next(
            e for e in mock_litellm.model_list if e["model_name"] == "local"
        )
        assert local_entry["litellm_params"]["api_base"] == "http://localhost:8801/v1"

        # Update the backend URL
        proxy.update_backend_url("mlx", 8899)

        # The entry should now reflect the new port
        assert local_entry["litellm_params"]["api_base"] == "http://localhost:8899/v1"


# ---------------------------------------------------------------------------
# Test 5: get_available_models returns all configured models
# ---------------------------------------------------------------------------

def test_get_available_models_returns_all():
    with patch("model_gateway.proxy.litellm") as mock_litellm:
        mock_litellm.model_list = []
        config = _make_config()
        proxy = ProxyManager(config)
        proxy.setup({"mlx": 8801})

        models = proxy.get_available_models()
        aliases = {m["alias"] for m in models}
        assert aliases == {"local", "haiku", "gpt4o"}


def test_get_available_models_includes_backend_info():
    with patch("model_gateway.proxy.litellm") as mock_litellm:
        mock_litellm.model_list = []
        config = _make_config()
        proxy = ProxyManager(config)
        proxy.setup({"mlx": 8801})

        models = proxy.get_available_models()
        local_info = next(m for m in models if m["alias"] == "local")
        assert local_info["backend"] == "mlx"
        assert local_info["model_id"] == "mlx-community/Qwen3-4B-4bit"
        assert local_info["api_base"] == "http://localhost:8801/v1"

        haiku_info = next(m for m in models if m["alias"] == "haiku")
        assert haiku_info["backend"] == "anthropic"
        # Cloud backends should not expose api_base
        assert "api_base" not in haiku_info


# ---------------------------------------------------------------------------
# Test 6: completion() calls litellm.acompletion with correct params (mock)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_completion_calls_litellm_acompletion():
    with patch("model_gateway.proxy.litellm") as mock_litellm:
        mock_litellm.model_list = []
        fake_response = MagicMock()
        mock_litellm.acompletion = AsyncMock(return_value=fake_response)

        config = _make_config()
        proxy = ProxyManager(config)
        proxy.setup({"mlx": 8801})

        messages = [{"role": "user", "content": "Hello"}]
        result = await proxy.completion("haiku", messages, stream=False)

        mock_litellm.acompletion.assert_called_once_with(
            model="haiku",
            messages=messages,
            stream=False,
        )
        assert result is fake_response


@pytest.mark.asyncio
async def test_completion_passes_extra_kwargs():
    with patch("model_gateway.proxy.litellm") as mock_litellm:
        mock_litellm.model_list = []
        fake_response = MagicMock()
        mock_litellm.acompletion = AsyncMock(return_value=fake_response)

        config = _make_config()
        proxy = ProxyManager(config)
        proxy.setup({"mlx": 8801})

        messages = [{"role": "user", "content": "Summarize this"}]
        result = await proxy.completion("local", messages, stream=True, temperature=0.7)

        mock_litellm.acompletion.assert_called_once_with(
            model="local",
            messages=messages,
            stream=True,
            temperature=0.7,
        )
        assert result is fake_response


# ---------------------------------------------------------------------------
# Test: local model with no port registered is omitted from model_list
# ---------------------------------------------------------------------------

def test_local_model_without_port_omitted():
    with patch("model_gateway.proxy.litellm") as mock_litellm:
        mock_litellm.model_list = []
        config = _make_config()
        proxy = ProxyManager(config)
        # Don't provide port for "mlx"
        proxy.setup({})

        model_list = mock_litellm.model_list
        aliases = [e["model_name"] for e in model_list]
        assert "local" not in aliases
        # Cloud backends should still be registered
        assert "haiku" in aliases
        assert "gpt4o" in aliases
