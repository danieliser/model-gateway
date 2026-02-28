"""Configuration loading and validation for model-gateway."""

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    backend: str
    model_id: str | None = None
    model_path: str | None = None
    api_key_env: str | None = None
    pin: bool = False
    idle_timeout: int | None = None


@dataclass
class BackendConfig:
    enabled: bool = True
    host: str | None = None
    binary: str | None = None
    api_key_env: str | None = None


@dataclass
class GatewayConfig:
    port: int = 8800
    default_model: str | None = None
    embedding_model: str | None = None
    idle_timeout: int = 900
    idle_check_interval: int = 30
    models: dict[str, ModelConfig] = field(default_factory=dict)
    backends: dict[str, BackendConfig] = field(default_factory=dict)
    task_routing: dict[str, str] = field(default_factory=dict)
    fallback_chain: list[str] = field(default_factory=list)


def load_config(path: str | None = None) -> GatewayConfig:
    """Load config from YAML file.

    Search order:
    1. Explicit path argument
    2. $MODEL_GATEWAY_CONFIG env var
    3. ~/.config/model-gateway/config.yml
    4. $PAOP_ROOT/config/model-gateway.yml (if PAOP_ROOT set)

    Raises FileNotFoundError if no config found.
    """
    candidates: list[Path] = []

    if path is not None:
        candidates.append(Path(path))
    else:
        env_path = os.environ.get("MODEL_GATEWAY_CONFIG")
        if env_path:
            candidates.append(Path(env_path))

        candidates.append(Path.home() / ".config" / "model-gateway" / "config.yml")

        paop_root = os.environ.get("PAOP_ROOT")
        if paop_root:
            candidates.append(Path(paop_root) / "config" / "model-gateway.yml")

    for candidate in candidates:
        if candidate.exists():
            return _parse_config(candidate)

    checked = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"No config file found. Checked: {checked}")


def _parse_config(path: Path) -> GatewayConfig:
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    models = {}
    for name, cfg in (data.get("models") or {}).items():
        # Filter to known ModelConfig fields to avoid TypeError on unknown keys
        known = {"backend", "model_id", "model_path", "api_key_env", "pin", "idle_timeout"}
        filtered = {k: v for k, v in cfg.items() if k in known}
        models[name] = ModelConfig(**filtered)

    backends = {
        name: BackendConfig(**cfg)
        for name, cfg in (data.get("backends") or {}).items()
    }

    return GatewayConfig(
        port=data.get("port", 8800),
        default_model=data.get("default_model"),
        embedding_model=data.get("embedding_model"),
        idle_timeout=data.get("idle_timeout", 900),
        idle_check_interval=data.get("idle_check_interval", 30),
        models=models,
        backends=backends,
        task_routing=data.get("task_routing") or {},
        fallback_chain=data.get("fallback_chain") or [],
    )


CLOUD_BACKENDS = {"anthropic", "openai"}


def validate_config(config: GatewayConfig) -> list[str]:
    """Validate config, return list of errors/warnings.

    Checks:
    - default_model references a model in the models dict
    - Each model references a backend in the backends dict
    - task_routing values reference models in the models dict
    - fallback_chain entries reference backends in the backends dict
    - Cloud backends have api_key_env set
    - Models using local backends have either model_id or model_path
    """
    errors: list[str] = []

    if config.default_model and config.default_model not in config.models:
        errors.append(
            f"default_model '{config.default_model}' not found in models"
        )

    for model_name, model in config.models.items():
        if model.backend not in config.backends:
            errors.append(
                f"model '{model_name}' references unknown backend '{model.backend}'"
            )
        if model.backend not in CLOUD_BACKENDS:
            if not model.model_id and not model.model_path:
                errors.append(
                    f"model '{model_name}' uses local backend '{model.backend}' "
                    "but has neither model_id nor model_path"
                )

    for task, model_name in config.task_routing.items():
        if model_name not in config.models:
            errors.append(
                f"task_routing '{task}' references unknown model '{model_name}'"
            )

    for backend_name in config.fallback_chain:
        if backend_name not in config.backends:
            errors.append(
                f"fallback_chain entry '{backend_name}' not found in backends"
            )

    for backend_name, backend in config.backends.items():
        if backend_name in CLOUD_BACKENDS and backend.enabled and not backend.api_key_env:
            errors.append(
                f"cloud backend '{backend_name}' is enabled but has no api_key_env set"
            )

    return errors


def get_config_dir() -> Path:
    """Return ~/.config/model-gateway/, creating if needed."""
    config_dir = Path.home() / ".config" / "model-gateway"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_log_dir() -> Path:
    """Return ~/.config/model-gateway/logs/, creating if needed."""
    log_dir = get_config_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def get_pid_file() -> Path:
    """Return ~/.config/model-gateway/gateway.pid"""
    return get_config_dir() / "gateway.pid"
