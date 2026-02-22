"""Tests for model_gateway.routing module."""

import pytest

from model_gateway.config import BackendConfig, GatewayConfig, ModelConfig
from model_gateway.routing import TaskRouter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_config(
    default_model: str | None = "local",
    task_routing: dict | None = None,
    fallback_chain: list | None = None,
) -> GatewayConfig:
    return GatewayConfig(
        port=8800,
        default_model=default_model,
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
        task_routing=task_routing or {"coding": "haiku", "quick": "local"},
        fallback_chain=fallback_chain or ["anthropic", "mlx"],
    )


# ---------------------------------------------------------------------------
# Test 1: Known alias resolves to itself
# ---------------------------------------------------------------------------

def test_known_alias_resolves_to_itself():
    router = TaskRouter(_make_config())
    assert router.resolve_model("haiku") == "haiku"
    assert router.resolve_model("local") == "local"
    assert router.resolve_model("gpt4o") == "gpt4o"


# ---------------------------------------------------------------------------
# Test 2: "auto" with X-Task-Type header resolves correctly
# ---------------------------------------------------------------------------

def test_auto_with_task_type_header_resolves():
    router = TaskRouter(_make_config())
    result = router.resolve_model("auto", headers={"X-Task-Type": "coding"})
    assert result == "haiku"


def test_auto_with_task_type_header_quick():
    router = TaskRouter(_make_config())
    result = router.resolve_model("auto", headers={"X-Task-Type": "quick"})
    assert result == "local"


# ---------------------------------------------------------------------------
# Test 3: "auto" without header uses default_model
# ---------------------------------------------------------------------------

def test_auto_without_header_uses_default_model():
    router = TaskRouter(_make_config(default_model="local"))
    result = router.resolve_model("auto", headers=None)
    assert result == "local"


def test_auto_with_empty_headers_uses_default_model():
    router = TaskRouter(_make_config(default_model="haiku"))
    result = router.resolve_model("auto", headers={})
    assert result == "haiku"


# ---------------------------------------------------------------------------
# Test 4: "auto" with unknown task type uses default_model
# ---------------------------------------------------------------------------

def test_auto_with_unknown_task_type_uses_default():
    router = TaskRouter(_make_config(default_model="local"))
    result = router.resolve_model("auto", headers={"X-Task-Type": "does-not-exist"})
    assert result == "local"


# ---------------------------------------------------------------------------
# Test 5: Unknown model raises ValueError with available models list
# ---------------------------------------------------------------------------

def test_unknown_model_raises_value_error():
    router = TaskRouter(_make_config())
    with pytest.raises(ValueError) as exc_info:
        router.resolve_model("nonexistent-model")
    error_msg = str(exc_info.value)
    assert "nonexistent-model" in error_msg
    # Should mention available models
    assert "haiku" in error_msg or "local" in error_msg


# ---------------------------------------------------------------------------
# Test 6: Fallback when backend unhealthy finds alternative
# ---------------------------------------------------------------------------

def test_fallback_finds_healthy_alternative():
    config = _make_config(
        default_model="local",
        fallback_chain=["anthropic", "mlx"],
    )
    router = TaskRouter(config)
    # mlx is down, anthropic is healthy
    health = {"mlx": False, "anthropic": True, "openai": True}
    result = router.resolve_model("local", backend_health=health)
    # Should fall back to a model using anthropic
    assert result == "haiku"


def test_fallback_skips_unhealthy_backends():
    config = _make_config(
        default_model="local",
        fallback_chain=["anthropic", "openai", "mlx"],
    )
    router = TaskRouter(config)
    # mlx and anthropic are down, openai is healthy
    health = {"mlx": False, "anthropic": False, "openai": True}
    result = router.resolve_model("local", backend_health=health)
    assert result == "gpt4o"


# ---------------------------------------------------------------------------
# Test 7: Fallback when all backends unhealthy raises error
# ---------------------------------------------------------------------------

def test_fallback_all_backends_unhealthy_raises():
    config = _make_config(
        default_model="local",
        fallback_chain=["anthropic", "mlx"],
    )
    router = TaskRouter(config)
    # Everything is down
    health = {"mlx": False, "anthropic": False, "openai": False}
    with pytest.raises(ValueError) as exc_info:
        router.resolve_model("local", backend_health=health)
    assert "unavailable" in str(exc_info.value).lower() or "fallback" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Test 8: get_task_types returns config map
# ---------------------------------------------------------------------------

def test_get_task_types_returns_config_map():
    config = _make_config(task_routing={"coding": "haiku", "quick": "local"})
    router = TaskRouter(config)
    task_types = router.get_task_types()
    assert task_types == {"coding": "haiku", "quick": "local"}


def test_get_task_types_returns_copy():
    config = _make_config(task_routing={"coding": "haiku"})
    router = TaskRouter(config)
    task_types = router.get_task_types()
    task_types["injected"] = "bad"
    # Original should be unaffected
    assert "injected" not in router.get_task_types()


# ---------------------------------------------------------------------------
# Test: healthy backend skips fallback
# ---------------------------------------------------------------------------

def test_healthy_backend_no_fallback():
    config = _make_config(default_model="local", fallback_chain=["anthropic"])
    router = TaskRouter(config)
    health = {"mlx": True, "anthropic": True}
    result = router.resolve_model("local", backend_health=health)
    assert result == "local"


# ---------------------------------------------------------------------------
# Test: auto with no default_model raises
# ---------------------------------------------------------------------------

def test_auto_with_no_default_model_raises():
    config = _make_config(default_model=None, task_routing={})
    router = TaskRouter(config)
    with pytest.raises(ValueError) as exc_info:
        router.resolve_model("auto", headers={"X-Task-Type": "unknown"})
    assert "default_model" in str(exc_info.value) or "auto" in str(exc_info.value)
