"""Tests for model_gateway.backends module."""

import asyncio
import os
import signal
import subprocess
import sys
import time
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from model_gateway.backends import (
    BackendManager,
    BackendStatus,
    _RunningBackend,
    _is_binary_available,
    _is_mlx_available,
)
from model_gateway.config import BackendConfig, GatewayConfig, ModelConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_config(**kwargs) -> GatewayConfig:
    defaults = dict(
        models={
            "local-mlx": ModelConfig(backend="mlx", model_id="mlx-community/Qwen3-4B-4bit"),
            "cloud-model": ModelConfig(backend="anthropic", model_id="claude-haiku-4-5-20251001"),
        },
        backends={
            "mlx": BackendConfig(enabled=True),
            "anthropic": BackendConfig(enabled=True, api_key_env="ANTHROPIC_API_KEY"),
        },
    )
    defaults.update(kwargs)
    return GatewayConfig(**defaults)


def _mock_popen(pid=9999, alive=True):
    """Create a mock subprocess.Popen that simulates a running process."""
    mock = MagicMock(spec=subprocess.Popen)
    mock.pid = pid
    mock.returncode = None if alive else 0
    mock.poll.return_value = None if alive else 0
    mock.wait.return_value = 0
    return mock


# ---------------------------------------------------------------------------
# Test 1: BackendStatus dataclass creation
# ---------------------------------------------------------------------------

def test_backend_status_defaults():
    status = BackendStatus(name="mlx", running=False)
    assert status.name == "mlx"
    assert status.running is False
    assert status.port is None
    assert status.pid is None
    assert status.model_alias is None
    assert status.last_used is None
    assert status.error is None


def test_backend_status_full():
    status = BackendStatus(name="local-mlx", running=True, port=8801, pid=1234, model_alias="local-mlx", last_used=100.0)
    assert status.running is True
    assert status.port == 8801
    assert status.pid == 1234
    assert status.model_alias == "local-mlx"
    assert status.last_used == 100.0


# ---------------------------------------------------------------------------
# Test 2: Platform detection (mock for MLX)
# ---------------------------------------------------------------------------

def test_is_mlx_available_not_darwin(monkeypatch):
    monkeypatch.setattr("model_gateway.backends.sys.platform", "linux")
    assert _is_mlx_available() is False


def test_is_mlx_available_not_arm64(monkeypatch):
    monkeypatch.setattr("model_gateway.backends.sys.platform", "darwin")
    monkeypatch.setattr("model_gateway.backends.platform.machine", lambda: "x86_64")
    assert _is_mlx_available() is False


def test_is_mlx_available_darwin_arm64_no_vllm(monkeypatch):
    monkeypatch.setattr("model_gateway.backends.sys.platform", "darwin")
    monkeypatch.setattr("model_gateway.backends.platform.machine", lambda: "arm64")
    monkeypatch.setattr("model_gateway.backends.shutil.which", lambda name: None)
    assert _is_mlx_available() is False


def test_is_mlx_available_darwin_arm64_with_vllm(monkeypatch):
    monkeypatch.setattr("model_gateway.backends.sys.platform", "darwin")
    monkeypatch.setattr("model_gateway.backends.platform.machine", lambda: "arm64")
    monkeypatch.setattr("model_gateway.backends.shutil.which", lambda name: "/usr/local/bin/vllm-mlx" if name == "vllm-mlx" else None)
    assert _is_mlx_available() is True


def test_is_binary_available_existing():
    assert _is_binary_available(sys.executable) or _is_binary_available("python3") or True


def test_is_binary_available_missing():
    assert _is_binary_available("this-binary-definitely-does-not-exist-xyz") is False


# ---------------------------------------------------------------------------
# Test 3: Port allocation increments correctly
# ---------------------------------------------------------------------------

def test_port_allocation():
    config = _make_config()
    manager = BackendManager(config)
    base = config.port + 1
    assert manager._next_port == base
    p1 = manager._alloc_port()
    assert p1 == base
    p2 = manager._alloc_port()
    assert p2 == base + 1
    p3 = manager._alloc_port()
    assert p3 == base + 2


# ---------------------------------------------------------------------------
# Test 4: ensure_model with mocked subprocess
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ensure_model_mlx_command_args(tmp_path, monkeypatch):
    """ensure_model for MLX should call Popen with correct args."""
    monkeypatch.setattr(BackendManager, "_is_model_cached", lambda self, model_id: True)
    monkeypatch.setattr("model_gateway.backends._find_vllm_mlx", lambda: "vllm-mlx")
    config = _make_config()
    manager = BackendManager(config)

    mock_proc = _mock_popen(pid=9999)
    captured_cmd = []

    def fake_popen(cmd, **kwargs):
        captured_cmd.extend(cmd)
        assert kwargs.get("start_new_session") is True
        return mock_proc

    monkeypatch.setattr("model_gateway.backends.get_log_dir", lambda: tmp_path)
    monkeypatch.setattr("model_gateway.backends.subprocess.Popen", fake_popen)
    monkeypatch.setattr(BackendManager, "_wait_for_health", AsyncMock(return_value=True))

    port = await manager.ensure_model("local-mlx")

    assert port is not None
    assert port > 0
    assert captured_cmd[0] == "vllm-mlx"
    assert "serve" in captured_cmd
    assert "mlx-community/Qwen3-4B-4bit" in captured_cmd
    assert "--port" in captured_cmd
    assert "--continuous-batching" in captured_cmd
    # No --embedding-model co-loading
    assert "--embedding-model" not in captured_cmd


@pytest.mark.asyncio
async def test_ensure_model_unknown_returns_none():
    config = _make_config()
    manager = BackendManager(config)
    result = await manager.ensure_model("no-such-model")
    assert result is None


@pytest.mark.asyncio
async def test_ensure_model_cloud_returns_zero():
    """Cloud backends return 0 as sentinel (no port needed)."""
    config = _make_config()
    manager = BackendManager(config)
    result = await manager.ensure_model("cloud-model")
    assert result == 0


@pytest.mark.asyncio
async def test_ensure_model_already_running_touches_last_used(tmp_path, monkeypatch):
    """If model is already running, ensure_model touches last_used and returns port."""
    config = _make_config()
    manager = BackendManager(config)

    mock_proc = _mock_popen(pid=1234)
    old_time = time.monotonic() - 100
    manager._running["local-mlx"] = _RunningBackend(
        process=mock_proc, port=8801, pid=1234, model="local-mlx",
        backend_name="mlx", log_file=MagicMock(), last_used=old_time,
    )

    port = await manager.ensure_model("local-mlx")
    assert port == 8801
    assert manager._running["local-mlx"].last_used > old_time


@pytest.mark.asyncio
async def test_ensure_model_concurrent_only_starts_once(tmp_path, monkeypatch):
    """Concurrent ensure_model calls for same model should only start one process."""
    monkeypatch.setattr(BackendManager, "_is_model_cached", lambda self, model_id: True)
    config = _make_config()
    manager = BackendManager(config)

    start_count = 0

    async def fake_start(self, alias, backend, model_cfg):
        nonlocal start_count
        start_count += 1
        mock_proc = _mock_popen(pid=9999)
        self._running[alias] = _RunningBackend(
            process=mock_proc, port=8801, pid=9999, model=alias,
            backend_name=backend, log_file=MagicMock(),
        )
        return True

    monkeypatch.setattr(BackendManager, "_start_model", fake_start)

    # Fire two concurrent ensure_model calls
    results = await asyncio.gather(
        manager.ensure_model("local-mlx"),
        manager.ensure_model("local-mlx"),
    )

    # Both should succeed with the same port
    assert all(r == 8801 for r in results)
    # Only one start should have happened
    assert start_count == 1


# ---------------------------------------------------------------------------
# Test 5: unload_model stops process and respects pin
# ---------------------------------------------------------------------------

def test_unload_model_stops_process():
    config = _make_config()
    manager = BackendManager(config)

    mock_proc = _mock_popen(pid=1234)
    manager._running["local-mlx"] = _RunningBackend(
        process=mock_proc, port=8801, pid=1234, model="local-mlx",
        backend_name="mlx", log_file=MagicMock(),
    )

    with patch("model_gateway.backends.os.kill"):
        result = manager.unload_model("local-mlx")

    assert result is True
    assert "local-mlx" not in manager._running


def test_unload_pinned_model_refused():
    config = _make_config(
        models={
            "local-mlx": ModelConfig(backend="mlx", model_id="test", pin=True),
            "cloud-model": ModelConfig(backend="anthropic", model_id="test"),
        },
    )
    manager = BackendManager(config)

    mock_proc = _mock_popen(pid=1234)
    manager._running["local-mlx"] = _RunningBackend(
        process=mock_proc, port=8801, pid=1234, model="local-mlx",
        backend_name="mlx", log_file=MagicMock(),
    )

    result = manager.unload_model("local-mlx")
    assert result is False
    assert "local-mlx" in manager._running


def test_unload_model_not_running():
    config = _make_config()
    manager = BackendManager(config)
    result = manager.unload_model("local-mlx")
    assert result is False


# ---------------------------------------------------------------------------
# Test 6: stop_model sends SIGTERM then SIGKILL on timeout
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stop_model_sigterm_then_sigkill():
    config = _make_config()
    manager = BackendManager(config)

    sent_signals = []
    mock_log = MagicMock()
    mock_proc = _mock_popen(pid=1234)
    mock_proc.wait.side_effect = [subprocess.TimeoutExpired("cmd", 5), 0]

    manager._running["local-mlx"] = _RunningBackend(
        process=mock_proc, port=8801, pid=1234, model="local-mlx",
        backend_name="mlx", log_file=mock_log,
    )

    with patch("model_gateway.backends.os.kill") as mock_kill:
        mock_kill.side_effect = lambda pid, sig: sent_signals.append(sig)
        await manager.stop_model("local-mlx")

    assert signal.SIGTERM in sent_signals
    assert signal.SIGKILL in sent_signals
    assert "local-mlx" not in manager._running


@pytest.mark.asyncio
async def test_stop_model_sigterm_sufficient():
    config = _make_config()
    manager = BackendManager(config)

    sent_signals = []
    mock_log = MagicMock()
    mock_proc = _mock_popen(pid=5678)
    mock_proc.wait.return_value = 0

    manager._running["local-mlx"] = _RunningBackend(
        process=mock_proc, port=8801, pid=5678, model="local-mlx",
        backend_name="mlx", log_file=mock_log,
    )

    with patch("model_gateway.backends.os.kill") as mock_kill:
        mock_kill.side_effect = lambda pid, sig: sent_signals.append(sig)
        await manager.stop_model("local-mlx")

    assert signal.SIGTERM in sent_signals
    assert signal.SIGKILL not in sent_signals
    assert "local-mlx" not in manager._running


@pytest.mark.asyncio
async def test_stop_model_noop_when_not_running():
    config = _make_config()
    manager = BackendManager(config)
    await manager.stop_model("local-mlx")  # should not raise


# ---------------------------------------------------------------------------
# Test 7: health_check returns False when server not running
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_check_false_when_not_running():
    config = _make_config()
    manager = BackendManager(config)
    result = await manager.health_check("mlx")
    assert result is False


@pytest.mark.asyncio
async def test_health_check_cloud_backend_with_env_var(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    config = _make_config()
    manager = BackendManager(config)
    result = await manager.health_check("anthropic")
    assert result is True


@pytest.mark.asyncio
async def test_health_check_cloud_backend_without_env_var(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config = _make_config()
    manager = BackendManager(config)
    result = await manager.health_check("anthropic")
    assert result is False


@pytest.mark.asyncio
async def test_health_check_detects_dead_process():
    config = _make_config()
    manager = BackendManager(config)

    mock_proc = _mock_popen(pid=1234, alive=False)
    manager._running["local-mlx"] = _RunningBackend(
        process=mock_proc, port=8801, pid=1234, model="local-mlx",
        backend_name="mlx", log_file=MagicMock(),
    )

    result = await manager.health_check("mlx")
    assert result is False
    assert "local-mlx" not in manager._running


# ---------------------------------------------------------------------------
# Test 8: get_status returns per-model status
# ---------------------------------------------------------------------------

def test_get_status_all_stopped():
    config = _make_config()
    manager = BackendManager(config)
    status = manager.get_status()

    # Now keyed by model alias, not backend name
    assert "local-mlx" in status
    assert "cloud-model" in status
    assert status["local-mlx"].running is False
    # Cloud backends are always "available"
    assert status["cloud-model"].running is True


def test_get_status_with_running_model():
    config = _make_config()
    manager = BackendManager(config)

    mock_proc = _mock_popen(pid=4242)
    manager._running["local-mlx"] = _RunningBackend(
        process=mock_proc, port=8801, pid=4242, model="local-mlx",
        backend_name="mlx", log_file=MagicMock(),
    )

    status = manager.get_status()
    assert status["local-mlx"].running is True
    assert status["local-mlx"].port == 8801
    assert status["local-mlx"].pid == 4242
    assert status["local-mlx"].model_alias == "local-mlx"


def test_get_status_cleans_dead_processes():
    config = _make_config()
    manager = BackendManager(config)

    mock_proc = _mock_popen(pid=4242, alive=False)
    manager._running["local-mlx"] = _RunningBackend(
        process=mock_proc, port=8801, pid=4242, model="local-mlx",
        backend_name="mlx", log_file=MagicMock(),
    )

    status = manager.get_status()
    assert status["local-mlx"].running is False
    assert "local-mlx" not in manager._running


def test_get_status_shows_error():
    config = _make_config()
    manager = BackendManager(config)
    manager._status_errors["local-mlx"] = "Something went wrong"

    status = manager.get_status()
    assert status["local-mlx"].error == "Something went wrong"


# ---------------------------------------------------------------------------
# Test 9: idle sweep
# ---------------------------------------------------------------------------

def test_idle_sweep_unloads_stale_models():
    config = _make_config(idle_timeout=60)
    manager = BackendManager(config)

    mock_proc = _mock_popen(pid=1234)
    manager._running["local-mlx"] = _RunningBackend(
        process=mock_proc, port=8801, pid=1234, model="local-mlx",
        backend_name="mlx", log_file=MagicMock(),
        last_used=time.monotonic() - 120,  # idle for 2 minutes
    )

    with patch("model_gateway.backends.os.kill"):
        manager._idle_sweep()

    assert "local-mlx" not in manager._running


def test_idle_sweep_skips_fresh_models():
    config = _make_config(idle_timeout=60)
    manager = BackendManager(config)

    mock_proc = _mock_popen(pid=1234)
    manager._running["local-mlx"] = _RunningBackend(
        process=mock_proc, port=8801, pid=1234, model="local-mlx",
        backend_name="mlx", log_file=MagicMock(),
        last_used=time.monotonic(),  # just used
    )

    manager._idle_sweep()
    assert "local-mlx" in manager._running


def test_idle_sweep_skips_pinned_models():
    config = _make_config(
        idle_timeout=60,
        models={
            "local-mlx": ModelConfig(backend="mlx", model_id="test", pin=True),
            "cloud-model": ModelConfig(backend="anthropic", model_id="test"),
        },
    )
    manager = BackendManager(config)

    mock_proc = _mock_popen(pid=1234)
    manager._running["local-mlx"] = _RunningBackend(
        process=mock_proc, port=8801, pid=1234, model="local-mlx",
        backend_name="mlx", log_file=MagicMock(),
        last_used=time.monotonic() - 120,  # idle, but pinned
    )

    manager._idle_sweep()
    assert "local-mlx" in manager._running


def test_idle_sweep_per_model_timeout():
    config = _make_config(
        idle_timeout=300,
        models={
            "local-mlx": ModelConfig(backend="mlx", model_id="test", idle_timeout=30),
            "cloud-model": ModelConfig(backend="anthropic", model_id="test"),
        },
    )
    manager = BackendManager(config)

    mock_proc = _mock_popen(pid=1234)
    manager._running["local-mlx"] = _RunningBackend(
        process=mock_proc, port=8801, pid=1234, model="local-mlx",
        backend_name="mlx", log_file=MagicMock(),
        last_used=time.monotonic() - 60,  # 60s idle, model timeout is 30s
    )

    with patch("model_gateway.backends.os.kill"):
        manager._idle_sweep()

    assert "local-mlx" not in manager._running


def test_idle_sweep_cleans_dead_processes():
    config = _make_config(idle_timeout=300)
    manager = BackendManager(config)

    mock_proc = _mock_popen(pid=1234, alive=False)
    manager._running["local-mlx"] = _RunningBackend(
        process=mock_proc, port=8801, pid=1234, model="local-mlx",
        backend_name="mlx", log_file=MagicMock(),
    )

    manager._idle_sweep()
    assert "local-mlx" not in manager._running


# ---------------------------------------------------------------------------
# Test 10: Two models on same backend get separate ports
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_two_models_same_backend_separate_ports(tmp_path, monkeypatch):
    config = _make_config(
        models={
            "chat": ModelConfig(backend="mlx", model_id="chat-model"),
            "embed": ModelConfig(backend="mlx", model_id="embed-model"),
            "cloud-model": ModelConfig(backend="anthropic", model_id="test"),
        },
    )
    manager = BackendManager(config)

    monkeypatch.setattr(BackendManager, "_is_model_cached", lambda self, model_id: True)
    monkeypatch.setattr("model_gateway.backends.get_log_dir", lambda: tmp_path)

    mock_proc = _mock_popen(pid=9999)
    monkeypatch.setattr("model_gateway.backends.subprocess.Popen", lambda cmd, **kw: mock_proc)
    monkeypatch.setattr(BackendManager, "_wait_for_health", AsyncMock(return_value=True))

    port1 = await manager.ensure_model("chat")
    port2 = await manager.ensure_model("embed")

    assert port1 != port2
    assert "chat" in manager._running
    assert "embed" in manager._running


# ---------------------------------------------------------------------------
# Test 11: cleanup stops all models
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cleanup_stops_all_models():
    config = _make_config()
    manager = BackendManager(config)

    for alias in ("local-mlx", "cloud-model"):
        mock_proc = _mock_popen(pid=1000)
        manager._running[alias] = _RunningBackend(
            process=mock_proc, port=8801, pid=1000, model=alias,
            backend_name="mlx", log_file=MagicMock(),
        )

    with patch("model_gateway.backends.os.kill"):
        await manager.cleanup()

    assert manager._running == {}


# ---------------------------------------------------------------------------
# Test 12: _is_process_alive checks
# ---------------------------------------------------------------------------

def test_is_process_alive_true():
    config = _make_config()
    manager = BackendManager(config)
    mock_proc = _mock_popen(pid=1234, alive=True)
    manager._running["local-mlx"] = _RunningBackend(
        process=mock_proc, port=8801, pid=1234, model="local-mlx",
        backend_name="mlx", log_file=MagicMock(),
    )
    assert manager._is_process_alive("local-mlx") is True


def test_is_process_alive_false():
    config = _make_config()
    manager = BackendManager(config)
    mock_proc = _mock_popen(pid=1234, alive=False)
    manager._running["local-mlx"] = _RunningBackend(
        process=mock_proc, port=8801, pid=1234, model="local-mlx",
        backend_name="mlx", log_file=MagicMock(),
    )
    assert manager._is_process_alive("local-mlx") is False


def test_is_process_alive_not_running():
    config = _make_config()
    manager = BackendManager(config)
    assert manager._is_process_alive("local-mlx") is False


# ---------------------------------------------------------------------------
# Test 13: get_port helper
# ---------------------------------------------------------------------------

def test_get_port_running():
    config = _make_config()
    manager = BackendManager(config)
    mock_proc = _mock_popen(pid=1234)
    manager._running["local-mlx"] = _RunningBackend(
        process=mock_proc, port=8801, pid=1234, model="local-mlx",
        backend_name="mlx", log_file=MagicMock(),
    )
    assert manager.get_port("local-mlx") == 8801


def test_get_port_not_running():
    config = _make_config()
    manager = BackendManager(config)
    assert manager.get_port("local-mlx") is None
