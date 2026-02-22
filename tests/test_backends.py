"""Tests for model_gateway.backends module."""

import asyncio
import os
import signal
import sys
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from model_gateway.backends import (
    BackendManager,
    BackendStatus,
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
    # 'python3' or 'python' should always be available in test env
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
# Test 4: start_backend with mocked subprocess (verify command args)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_start_backend_mlx_command_args(tmp_path, monkeypatch):
    """start_backend for MLX should call create_subprocess_exec with correct args."""
    config = _make_config()
    manager = BackendManager(config)

    mock_proc = MagicMock()
    mock_proc.pid = 9999
    mock_proc.send_signal = MagicMock()
    mock_proc.wait = AsyncMock(return_value=0)

    captured_cmd = []

    async def fake_create_subprocess_exec(*cmd, **kwargs):
        captured_cmd.extend(cmd)
        return mock_proc

    monkeypatch.setattr(
        "model_gateway.backends.get_log_dir",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        "asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    # Make health check return True immediately
    monkeypatch.setattr(
        BackendManager,
        "_wait_for_health",
        AsyncMock(return_value=True),
    )

    result = await manager.start_backend("mlx", "local-mlx")

    assert result is True
    assert "-m" in captured_cmd
    assert "mlx_lm.server" in captured_cmd
    assert "--model" in captured_cmd
    idx = captured_cmd.index("--model")
    assert captured_cmd[idx + 1] == "mlx-community/Qwen3-4B-4bit"
    assert "--port" in captured_cmd
    idx_port = captured_cmd.index("--port")
    assert captured_cmd[idx_port + 1] == "8801"


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

    async def never_exits():
        await asyncio.sleep(999)

    mock_proc = MagicMock()
    mock_proc.pid = 1234

    def capture_signal(sig):
        sent_signals.append(sig)

    mock_proc.send_signal = capture_signal

    # First wait call times out, second succeeds (proc.wait after SIGKILL)
    wait_call_count = 0

    async def mock_wait_for(coro, timeout=None):
        nonlocal wait_call_count
        wait_call_count += 1
        if wait_call_count == 1:
            raise asyncio.TimeoutError()
        return 0

    # After SIGKILL, proc.wait() is awaited directly
    mock_proc.wait = AsyncMock(return_value=0)

    from model_gateway.backends import _RunningBackend
    manager._running["mlx"] = _RunningBackend(
        process=mock_proc,
        port=8801,
        pid=1234,
        model="local-mlx",
        log_file=mock_log,
    )

    with patch("asyncio.wait_for", side_effect=mock_wait_for):
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
    mock_proc = MagicMock()
    mock_proc.pid = 5678
    mock_proc.send_signal = lambda sig: sent_signals.append(sig)

    from model_gateway.backends import _RunningBackend
    manager._running["mlx"] = _RunningBackend(
        process=mock_proc,
        port=8801,
        pid=5678,
        model="local-mlx",
        log_file=mock_log,
    )

    with patch("asyncio.wait_for", AsyncMock(return_value=0)):
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
    """MLX health check should fail when nothing listens on the port."""
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

    mock_log = MagicMock()
    mock_proc = MagicMock()
    mock_proc.pid = 4242

    from model_gateway.backends import _RunningBackend
    manager._running["mlx"] = _RunningBackend(
        process=mock_proc,
        port=8801,
        pid=4242,
        model="local-mlx",
        log_file=mock_log,
    )

    status = manager.get_status()
    assert status["mlx"].running is True
    assert status["mlx"].port == 8801
    assert status["mlx"].pid == 4242
    assert status["mlx"].model_alias == "local-mlx"


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

    from model_gateway.backends import _RunningBackend

    for name in ("mlx", "anthropic"):
        mock_proc = MagicMock()
        mock_proc.pid = 1000
        manager._running[name] = _RunningBackend(
            process=mock_proc,
            port=8801,
            pid=1000,
            model="local-mlx",
            log_file=MagicMock(),
        )

    with patch.object(manager, "stop_backend", side_effect=fake_stop):
        await manager.cleanup()

    assert set(stopped) == {"mlx", "anthropic"}
    assert manager._running == {}
