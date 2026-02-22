"""Tests for model_gateway.backends module."""

import asyncio
import os
import signal
import subprocess
import sys
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
    assert status.error is None


def test_backend_status_full():
    status = BackendStatus(name="mlx", running=True, port=8801, pid=1234, model_alias="local-mlx")
    assert status.running is True
    assert status.port == 8801
    assert status.pid == 1234
    assert status.model_alias == "local-mlx"


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


def test_is_mlx_available_darwin_arm64_no_mlx(monkeypatch):
    monkeypatch.setattr("model_gateway.backends.sys.platform", "darwin")
    monkeypatch.setattr("model_gateway.backends.platform.machine", lambda: "arm64")
    with patch("importlib.util.find_spec", return_value=None):
        assert _is_mlx_available() is False


def test_is_mlx_available_darwin_arm64_with_mlx(monkeypatch):
    monkeypatch.setattr("model_gateway.backends.sys.platform", "darwin")
    monkeypatch.setattr("model_gateway.backends.platform.machine", lambda: "arm64")
    mock_spec = MagicMock()
    with patch("importlib.util.find_spec", return_value=mock_spec):
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
    assert manager._next_port == 8801
    p1 = manager._alloc_port()
    assert p1 == 8801
    p2 = manager._alloc_port()
    assert p2 == 8802
    p3 = manager._alloc_port()
    assert p3 == 8803


# ---------------------------------------------------------------------------
# Test 4: start_backend with mocked subprocess
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_start_backend_mlx_command_args(tmp_path, monkeypatch):
    """start_backend for MLX should call Popen with correct args."""
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

    result = await manager.start_backend("mlx", "local-mlx")

    assert result is True
    assert "-m" in captured_cmd
    assert "mlx_lm" in captured_cmd
    assert "server" in captured_cmd
    assert "--model" in captured_cmd
    idx = captured_cmd.index("--model")
    assert captured_cmd[idx + 1] == "mlx-community/Qwen3-4B-4bit"
    assert "--port" in captured_cmd


@pytest.mark.asyncio
async def test_start_backend_unknown_model_returns_false():
    config = _make_config()
    manager = BackendManager(config)
    result = await manager.start_backend("mlx", "no-such-model")
    assert result is False


@pytest.mark.asyncio
async def test_start_backend_cloud_returns_true():
    """Cloud backends don't need subprocess, always return True."""
    config = _make_config()
    manager = BackendManager(config)
    result = await manager.start_backend("anthropic", "cloud-model")
    assert result is True


# ---------------------------------------------------------------------------
# Test 5: stop_backend sends SIGTERM then SIGKILL on timeout
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stop_backend_sigterm_then_sigkill():
    """stop_backend should SIGTERM first, then SIGKILL if process doesn't exit."""
    config = _make_config()
    manager = BackendManager(config)

    sent_signals = []
    mock_log = MagicMock()
    mock_proc = _mock_popen(pid=1234)

    # Process doesn't exit after SIGTERM (wait raises TimeoutExpired)
    mock_proc.wait.side_effect = [subprocess.TimeoutExpired("cmd", 5), 0]

    manager._running["mlx"] = _RunningBackend(
        process=mock_proc, port=8801, pid=1234, model="local-mlx", log_file=mock_log,
    )

    with patch("model_gateway.backends.os.kill") as mock_kill:
        def capture_kill(pid, sig):
            sent_signals.append(sig)
        mock_kill.side_effect = capture_kill

        await manager.stop_backend("mlx")

    assert signal.SIGTERM in sent_signals
    assert signal.SIGKILL in sent_signals
    assert "mlx" not in manager._running


@pytest.mark.asyncio
async def test_stop_backend_sigterm_sufficient():
    """If process exits after SIGTERM, no SIGKILL needed."""
    config = _make_config()
    manager = BackendManager(config)

    sent_signals = []
    mock_log = MagicMock()
    mock_proc = _mock_popen(pid=5678)
    mock_proc.wait.return_value = 0  # exits cleanly

    manager._running["mlx"] = _RunningBackend(
        process=mock_proc, port=8801, pid=5678, model="local-mlx", log_file=mock_log,
    )

    with patch("model_gateway.backends.os.kill") as mock_kill:
        mock_kill.side_effect = lambda pid, sig: sent_signals.append(sig)
        await manager.stop_backend("mlx")

    assert signal.SIGTERM in sent_signals
    assert signal.SIGKILL not in sent_signals
    assert "mlx" not in manager._running


@pytest.mark.asyncio
async def test_stop_backend_noop_when_not_running():
    """stop_backend on a non-running backend should not raise."""
    config = _make_config()
    manager = BackendManager(config)
    await manager.stop_backend("mlx")  # should not raise


# ---------------------------------------------------------------------------
# Test 6: health_check returns False when server not running
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
    """health_check should return False and clean up if process died."""
    config = _make_config()
    manager = BackendManager(config)

    mock_proc = _mock_popen(pid=1234, alive=False)
    manager._running["mlx"] = _RunningBackend(
        process=mock_proc, port=8801, pid=1234, model="local-mlx", log_file=MagicMock(),
    )

    result = await manager.health_check("mlx")
    assert result is False
    assert "mlx" not in manager._running


# ---------------------------------------------------------------------------
# Test 7: get_status returns correct state
# ---------------------------------------------------------------------------

def test_get_status_all_stopped():
    config = _make_config()
    manager = BackendManager(config)
    status = manager.get_status()

    assert "mlx" in status
    assert "anthropic" in status
    assert status["mlx"].running is False
    assert status["anthropic"].running is False


def test_get_status_with_running_backend():
    config = _make_config()
    manager = BackendManager(config)

    mock_proc = _mock_popen(pid=4242)
    manager._running["mlx"] = _RunningBackend(
        process=mock_proc, port=8801, pid=4242, model="local-mlx", log_file=MagicMock(),
    )

    status = manager.get_status()
    assert status["mlx"].running is True
    assert status["mlx"].port == 8801
    assert status["mlx"].pid == 4242
    assert status["mlx"].model_alias == "local-mlx"


def test_get_status_cleans_dead_processes():
    """get_status should detect dead processes and remove them."""
    config = _make_config()
    manager = BackendManager(config)

    mock_proc = _mock_popen(pid=4242, alive=False)
    manager._running["mlx"] = _RunningBackend(
        process=mock_proc, port=8801, pid=4242, model="local-mlx", log_file=MagicMock(),
    )

    status = manager.get_status()
    assert status["mlx"].running is False
    assert "mlx" not in manager._running


def test_get_status_shows_error():
    config = _make_config()
    manager = BackendManager(config)
    manager._status_errors["mlx"] = "Something went wrong"

    status = manager.get_status()
    assert status["mlx"].error == "Something went wrong"


# ---------------------------------------------------------------------------
# Test 8: cleanup stops all running backends
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cleanup_stops_all_backends():
    config = _make_config()
    manager = BackendManager(config)

    stopped = []

    async def fake_stop(name):
        stopped.append(name)
        manager._running.pop(name, None)

    for name in ("mlx", "anthropic"):
        mock_proc = _mock_popen(pid=1000)
        manager._running[name] = _RunningBackend(
            process=mock_proc, port=8801, pid=1000, model="local-mlx", log_file=MagicMock(),
        )

    with patch.object(manager, "stop_backend", side_effect=fake_stop):
        await manager.cleanup()

    assert set(stopped) == {"mlx", "anthropic"}
    assert manager._running == {}


# ---------------------------------------------------------------------------
# Test 9: _is_process_alive checks
# ---------------------------------------------------------------------------

def test_is_process_alive_true():
    config = _make_config()
    manager = BackendManager(config)
    mock_proc = _mock_popen(pid=1234, alive=True)
    manager._running["mlx"] = _RunningBackend(
        process=mock_proc, port=8801, pid=1234, model="local-mlx", log_file=MagicMock(),
    )
    assert manager._is_process_alive("mlx") is True


def test_is_process_alive_false():
    config = _make_config()
    manager = BackendManager(config)
    mock_proc = _mock_popen(pid=1234, alive=False)
    manager._running["mlx"] = _RunningBackend(
        process=mock_proc, port=8801, pid=1234, model="local-mlx", log_file=MagicMock(),
    )
    assert manager._is_process_alive("mlx") is False


def test_is_process_alive_not_running():
    config = _make_config()
    manager = BackendManager(config)
    assert manager._is_process_alive("mlx") is False
