"""Tests for model_gateway.config module."""

import os
import pytest
import yaml

from model_gateway.config import (
    GatewayConfig,
    ModelConfig,
    BackendConfig,
    load_config,
    validate_config,
    get_config_dir,
    get_log_dir,
    get_pid_file,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(path, data):
    path.write_text(yaml.dump(data))
    return path


MINIMAL_VALID_DATA = {
    "port": 8800,
    "default_model": "local",
    "models": {
        "local": {
            "backend": "mlx",
            "model_id": "mlx-community/Qwen3-4B-4bit",
        },
        "cloud": {
            "backend": "anthropic",
            "model_id": "claude-haiku-4-5-20251001",
            "api_key_env": "ANTHROPIC_API_KEY",
        },
    },
    "backends": {
        "mlx": {"enabled": True},
        "anthropic": {"enabled": True, "api_key_env": "ANTHROPIC_API_KEY"},
    },
    "task_routing": {
        "coding": "cloud",
    },
    "fallback_chain": ["mlx"],
}


# ---------------------------------------------------------------------------
# Test 1: Load valid config from file
# ---------------------------------------------------------------------------

def test_load_valid_config(tmp_path):
    config_file = _write_yaml(tmp_path / "config.yml", MINIMAL_VALID_DATA)
    config = load_config(str(config_file))

    assert config.port == 8800
    assert config.default_model == "local"
    assert "local" in config.models
    assert config.models["local"].backend == "mlx"
    assert config.models["local"].model_id == "mlx-community/Qwen3-4B-4bit"
    assert "anthropic" in config.backends
    assert config.backends["anthropic"].api_key_env == "ANTHROPIC_API_KEY"
    assert config.task_routing["coding"] == "cloud"
    assert config.fallback_chain == ["mlx"]


# ---------------------------------------------------------------------------
# Test 2: Missing config file raises FileNotFoundError
# ---------------------------------------------------------------------------

def test_load_missing_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "nonexistent.yml"))


# ---------------------------------------------------------------------------
# Test 3: Empty config file gets defaults
# ---------------------------------------------------------------------------

def test_empty_config_gets_defaults(tmp_path):
    config_file = tmp_path / "config.yml"
    config_file.write_text("")  # empty YAML -> None -> {}
    config = load_config(str(config_file))

    assert config.port == 8800
    assert config.default_model is None
    assert config.models == {}
    assert config.backends == {}
    assert config.task_routing == {}
    assert config.fallback_chain == []


# ---------------------------------------------------------------------------
# Test 4: Invalid backend reference in model detected
# ---------------------------------------------------------------------------

def test_validate_invalid_backend_in_model():
    config = GatewayConfig(
        models={"m": ModelConfig(backend="missing-backend", model_id="x")},
        backends={},
    )
    errors = validate_config(config)
    assert any("missing-backend" in e for e in errors)


# ---------------------------------------------------------------------------
# Test 5: Invalid model reference in task_routing detected
# ---------------------------------------------------------------------------

def test_validate_invalid_model_in_task_routing():
    config = GatewayConfig(
        models={},
        backends={},
        task_routing={"coding": "ghost-model"},
    )
    errors = validate_config(config)
    assert any("ghost-model" in e for e in errors)


# ---------------------------------------------------------------------------
# Test 6: default_model not in models detected
# ---------------------------------------------------------------------------

def test_validate_default_model_not_in_models():
    config = GatewayConfig(
        default_model="nonexistent",
        models={},
        backends={},
    )
    errors = validate_config(config)
    assert any("default_model" in e and "nonexistent" in e for e in errors)


# ---------------------------------------------------------------------------
# Test 7: Cloud backend missing api_key_env generates warning
# ---------------------------------------------------------------------------

def test_validate_cloud_backend_missing_api_key_env():
    config = GatewayConfig(
        models={},
        backends={"anthropic": BackendConfig(enabled=True, api_key_env=None)},
    )
    errors = validate_config(config)
    assert any("anthropic" in e and "api_key_env" in e for e in errors)


def test_validate_cloud_backend_disabled_no_warning():
    """Disabled cloud backend without api_key_env should NOT warn."""
    config = GatewayConfig(
        models={},
        backends={"anthropic": BackendConfig(enabled=False, api_key_env=None)},
    )
    errors = validate_config(config)
    assert not any("anthropic" in e and "api_key_env" in e for e in errors)


# ---------------------------------------------------------------------------
# Test 8: Config search path uses env var override
# ---------------------------------------------------------------------------

def test_load_config_env_var_override(tmp_path, monkeypatch):
    config_file = _write_yaml(tmp_path / "custom.yml", MINIMAL_VALID_DATA)
    monkeypatch.setenv("MODEL_GATEWAY_CONFIG", str(config_file))
    monkeypatch.delenv("PAOP_ROOT", raising=False)

    config = load_config()  # no explicit path
    assert config.default_model == "local"


def test_load_config_no_env_var_no_default_raises(tmp_path, monkeypatch):
    """Without env var and without a real ~/.config file, should raise."""
    monkeypatch.delenv("MODEL_GATEWAY_CONFIG", raising=False)
    monkeypatch.delenv("PAOP_ROOT", raising=False)
    # Override home so ~/.config/model-gateway/config.yml won't exist
    monkeypatch.setattr(
        "model_gateway.config.Path.home",
        lambda: tmp_path,
    )
    with pytest.raises(FileNotFoundError):
        load_config()


# ---------------------------------------------------------------------------
# Bonus: validate_config passes on clean config
# ---------------------------------------------------------------------------

def test_validate_clean_config_no_errors(tmp_path):
    config_file = _write_yaml(tmp_path / "config.yml", MINIMAL_VALID_DATA)
    config = load_config(str(config_file))
    errors = validate_config(config)
    assert errors == []


# ---------------------------------------------------------------------------
# Bonus: fallback_chain entry not in backends detected
# ---------------------------------------------------------------------------

def test_validate_fallback_chain_unknown_backend():
    config = GatewayConfig(
        models={},
        backends={},
        fallback_chain=["does-not-exist"],
    )
    errors = validate_config(config)
    assert any("does-not-exist" in e for e in errors)


# ---------------------------------------------------------------------------
# Bonus: local model without model_id or model_path detected
# ---------------------------------------------------------------------------

def test_validate_local_model_needs_model_id_or_path():
    config = GatewayConfig(
        models={"m": ModelConfig(backend="mlx")},
        backends={"mlx": BackendConfig()},
    )
    errors = validate_config(config)
    assert any("model_id" in e or "model_path" in e for e in errors)


def test_validate_local_model_with_model_id_ok():
    config = GatewayConfig(
        models={"m": ModelConfig(backend="mlx", model_id="some/model")},
        backends={"mlx": BackendConfig()},
    )
    errors = validate_config(config)
    assert errors == []
