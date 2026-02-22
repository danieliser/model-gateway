"""Task-based model routing — maps task types to model aliases and handles fallback chains."""

import logging

from model_gateway.config import GatewayConfig

logger = logging.getLogger(__name__)

_CLOUD_BACKENDS = {"anthropic", "openai"}


class TaskRouter:
    def __init__(self, config: GatewayConfig):
        """Load task_routing and fallback_chain from config."""
        self._config = config

    def resolve_model(
        self,
        requested_model: str,
        headers: dict | None = None,
        backend_health: dict[str, bool] | None = None,
    ) -> str:
        """Resolve requested model to actual model alias.

        Logic:
        1. If model == "auto": look up X-Task-Type header in task_routing config
           - If no header or unknown task type: use default_model
        2. If model is a known alias: use as-is
        3. If model is unknown: raise ValueError with available models

        Fallback:
        - If backend_health provided and model's backend is unhealthy:
          walk fallback_chain, find first model using a healthy backend
        - Log: "Model X unavailable (backend Y down), falling back to Z"
        """
        if requested_model == "auto":
            model = self._resolve_auto(headers)
        elif requested_model in self._config.models:
            model = requested_model
        else:
            available = sorted(self._config.models.keys())
            raise ValueError(
                f"Unknown model '{requested_model}'. "
                f"Available models: {available}"
            )

        if backend_health is not None:
            model = self._apply_fallback(model, backend_health)

        return model

    def _resolve_auto(self, headers: dict | None) -> str:
        """Resolve 'auto' using X-Task-Type header or fall back to default."""
        task_type = None
        if headers:
            task_type = headers.get("X-Task-Type") or headers.get("x-task-type")

        if task_type and task_type in self._config.task_routing:
            return self._config.task_routing[task_type]

        if self._config.default_model:
            return self._config.default_model

        raise ValueError(
            "Model 'auto' requested but no X-Task-Type header matched and no default_model configured"
        )

    def _apply_fallback(self, model: str, backend_health: dict[str, bool]) -> str:
        """Walk fallback_chain if the model's backend is unhealthy."""
        model_cfg = self._config.models.get(model)
        if model_cfg is None:
            return model

        backend = model_cfg.backend
        if backend_health.get(backend, True):
            return model

        # Backend is unhealthy — walk fallback_chain to find a healthy alternative
        for fallback_backend in self._config.fallback_chain:
            if fallback_backend == backend:
                continue
            if not backend_health.get(fallback_backend, True):
                continue
            # Find a model that uses this fallback backend
            for alias, cfg in self._config.models.items():
                if cfg.backend == fallback_backend:
                    logger.warning(
                        "Model %s unavailable (backend %s down), falling back to %s",
                        model,
                        backend,
                        alias,
                    )
                    return alias

        raise ValueError(
            f"Model '{model}' unavailable (backend '{backend}' down) "
            "and no healthy fallback found in fallback_chain"
        )

    def get_task_types(self) -> dict[str, str]:
        """Return the task_routing map for documentation."""
        return dict(self._config.task_routing)
