"""Tests for the model-gateway CLI."""

import os
import signal
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from click.testing import CliRunner

from model_gateway.cli import cli
from model_gateway.config import GatewayConfig, ModelConfig, BackendConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def minimal_config():
    return GatewayConfig(
        port=8800,
        default_model=None,
        models={},
        backends={},
    )


@pytest.fixture
def valid_config():
    return GatewayConfig(
        port=8800,
        default_model="qwen3",
        models={
            "qwen3": ModelConfig(backend="mlx", model_id="mlx-community/Qwen3-4B-4bit"),
        },
        backends={
            "mlx": BackendConfig(enabled=True),
        },
    )


# ---------------------------------------------------------------------------
# 1. --help shows all commands
# ---------------------------------------------------------------------------

def test_help_shows_all_commands(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    for cmd in ["start", "stop", "restart", "status", "models", "switch", "test", "config", "logs", "health"]:
        assert cmd in result.output


# ---------------------------------------------------------------------------
# 2. gateway start — mock subprocess, verify PID file created
# ---------------------------------------------------------------------------

def test_start_creates_pid_file(runner, tmp_path, minimal_config):
    pid_file = tmp_path / "gateway.pid"
    log_file = tmp_path / "gateway.log"

    mock_proc = MagicMock()
    mock_proc.pid = 12345

    with (
        patch("model_gateway.cli.load_config", return_value=minimal_config),
        patch("model_gateway.cli.get_pid_file", return_value=pid_file),
        patch("model_gateway.cli.get_log_dir", return_value=tmp_path),
        patch("model_gateway.cli.subprocess.Popen", return_value=mock_proc) as mock_popen,
        patch("model_gateway.cli._wait_for_health", return_value=True),
        patch("builtins.open", MagicMock()),
    ):
        result = runner.invoke(cli, ["start"])

    assert result.exit_code == 0, result.output
    assert pid_file.exists()
    assert pid_file.read_text() == "12345"
    assert "started" in result.output


def test_start_already_running(runner, tmp_path, minimal_config):
    pid_file = tmp_path / "gateway.pid"
    pid_file.write_text("99999")

    with (
        patch("model_gateway.cli.load_config", return_value=minimal_config),
        patch("model_gateway.cli.get_pid_file", return_value=pid_file),
        patch("model_gateway.cli._is_process_running", return_value=True),
    ):
        result = runner.invoke(cli, ["start"])

    assert result.exit_code == 0
    assert "already running" in result.output


def test_start_removes_stale_pid_file(runner, tmp_path, minimal_config):
    pid_file = tmp_path / "gateway.pid"
    pid_file.write_text("99999")

    mock_proc = MagicMock()
    mock_proc.pid = 22222

    with (
        patch("model_gateway.cli.load_config", return_value=minimal_config),
        patch("model_gateway.cli.get_pid_file", return_value=pid_file),
        patch("model_gateway.cli.get_log_dir", return_value=tmp_path),
        patch("model_gateway.cli._is_process_running", return_value=False),
        patch("model_gateway.cli.subprocess.Popen", return_value=mock_proc),
        patch("model_gateway.cli._wait_for_health", return_value=True),
        patch("builtins.open", MagicMock()),
    ):
        result = runner.invoke(cli, ["start"])

    assert result.exit_code == 0
    assert pid_file.read_text() == "22222"


# ---------------------------------------------------------------------------
# 3. gateway stop — verify PID file cleanup
# ---------------------------------------------------------------------------

def test_stop_cleans_up_pid_file(runner, tmp_path):
    pid_file = tmp_path / "gateway.pid"
    pid_file.write_text("55555")

    with (
        patch("model_gateway.cli.get_pid_file", return_value=pid_file),
        patch("model_gateway.cli.os.kill") as mock_kill,
        patch("model_gateway.cli._is_process_running", return_value=False),
        patch("model_gateway.cli.time.sleep"),
    ):
        result = runner.invoke(cli, ["stop"])

    assert result.exit_code == 0
    assert not pid_file.exists()
    mock_kill.assert_called_once_with(55555, signal.SIGTERM)
    assert "stopped" in result.output


def test_stop_when_not_running(runner, tmp_path):
    pid_file = tmp_path / "gateway.pid"
    # pid_file does not exist

    with patch("model_gateway.cli.get_pid_file", return_value=pid_file):
        result = runner.invoke(cli, ["stop"])

    assert result.exit_code == 0
    assert "not running" in result.output


def test_stop_handles_missing_process(runner, tmp_path):
    pid_file = tmp_path / "gateway.pid"
    pid_file.write_text("77777")

    with (
        patch("model_gateway.cli.get_pid_file", return_value=pid_file),
        patch("model_gateway.cli.os.kill", side_effect=ProcessLookupError),
        patch("model_gateway.cli.time.sleep"),
    ):
        result = runner.invoke(cli, ["stop"])

    assert result.exit_code == 0
    assert not pid_file.exists()


# ---------------------------------------------------------------------------
# 4. gateway status when not running
# ---------------------------------------------------------------------------

def test_status_not_running_no_pid_file(runner, tmp_path):
    pid_file = tmp_path / "gateway.pid"

    with patch("model_gateway.cli.get_pid_file", return_value=pid_file):
        result = runner.invoke(cli, ["status"])

    assert result.exit_code == 0
    assert "not running" in result.output


def test_status_stale_pid_file(runner, tmp_path):
    pid_file = tmp_path / "gateway.pid"
    pid_file.write_text("88888")

    with (
        patch("model_gateway.cli.get_pid_file", return_value=pid_file),
        patch("model_gateway.cli._is_process_running", return_value=False),
    ):
        result = runner.invoke(cli, ["status"])

    assert result.exit_code == 0
    assert "not running" in result.output
    assert not pid_file.exists()


def test_status_running_gateway(runner, tmp_path, minimal_config):
    pid_file = tmp_path / "gateway.pid"
    pid_file.write_text("33333")

    health_data = {
        "status": "ok",
        "uptime_seconds": 9240,
        "active_backend": "mlx_lm.server",
        "active_model": "mlx-community/Qwen3-4B",
        "backends": {
            "mlx": {"available": True, "reason": "mlx_lm.server in PATH"},
            "anthropic": {"available": True, "reason": "ANTHROPIC_API_KEY set"},
        },
    }

    with (
        patch("model_gateway.cli.get_pid_file", return_value=pid_file),
        patch("model_gateway.cli._is_process_running", return_value=True),
        patch("model_gateway.cli.load_config", return_value=minimal_config),
        patch("model_gateway.cli._gateway_request", return_value=health_data),
    ):
        result = runner.invoke(cli, ["status"])

    assert result.exit_code == 0
    assert "running" in result.output
    assert "33333" in result.output
    assert "2h 34m" in result.output


# ---------------------------------------------------------------------------
# 5. gateway config validate
# ---------------------------------------------------------------------------

def test_config_validate_valid(runner, valid_config):
    with patch("model_gateway.cli.load_config", return_value=valid_config):
        result = runner.invoke(cli, ["config", "validate"])

    assert result.exit_code == 0
    assert "valid" in result.output.lower()


def test_config_validate_with_errors(runner):
    bad_config = GatewayConfig(
        port=8800,
        default_model="missing-model",
        models={},
        backends={},
    )
    with patch("model_gateway.cli.load_config", return_value=bad_config):
        result = runner.invoke(cli, ["config", "validate"])

    assert result.exit_code != 0
    assert "missing-model" in result.output


def test_config_validate_no_config(runner):
    with patch("model_gateway.cli.load_config", side_effect=FileNotFoundError("No config found")):
        result = runner.invoke(cli, ["config", "validate"])

    assert result.exit_code != 0
    assert "No config found" in result.output
