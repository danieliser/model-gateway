"""Tests for the model-gateway CLI."""

import os
import signal
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import click
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
    for cmd in ["start", "stop", "restart", "status", "models", "switch", "test", "config", "logs", "health", "chat", "embed", "complete"]:
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


# ---------------------------------------------------------------------------
# 6. Direct-path helpers
# ---------------------------------------------------------------------------

def test_resolve_api_base_unknown_model(valid_config):
    from model_gateway.cli import _resolve_api_base

    with pytest.raises(click.ClickException, match="Unknown model"):
        _resolve_api_base(valid_config, "nonexistent")


def test_resolve_api_base_local_goes_direct(valid_config):
    """Local models resolve direct to backend port, not gateway."""
    from model_gateway.cli import _resolve_api_base

    url, model_id = _resolve_api_base(valid_config, "qwen3")

    assert url == "http://localhost:8801/v1"
    assert model_id == "mlx-community/Qwen3-4B-4bit"


def test_resolve_api_base_cloud_goes_through_gateway():
    """Cloud models route through the gateway for auth handling."""
    from model_gateway.cli import _resolve_api_base

    config = GatewayConfig(
        port=8800,
        default_model="haiku",
        models={"haiku": ModelConfig(backend="anthropic", model_id="claude-haiku-4-5-20251001")},
        backends={"anthropic": BackendConfig(enabled=True)},
    )
    url, model = _resolve_api_base(config, "haiku")
    assert url == "http://localhost:8800/v1"
    assert model == "haiku"


def test_default_model_from_config(valid_config):
    from model_gateway.cli import _default_model

    assert _default_model(valid_config) == "qwen3"


def test_default_model_no_models():
    from model_gateway.cli import _default_model

    config = GatewayConfig(port=8800, default_model=None, models={}, backends={})
    with pytest.raises(click.ClickException, match="No models configured"):
        _default_model(config)


def test_get_prompt_or_stdin_from_arg():
    from model_gateway.cli import _get_prompt_or_stdin

    assert _get_prompt_or_stdin("hello") == "hello"


def test_get_prompt_or_stdin_missing():
    from model_gateway.cli import _get_prompt_or_stdin

    with (
        patch("model_gateway.cli.sys.stdin") as mock_stdin,
    ):
        mock_stdin.isatty.return_value = True
        with pytest.raises(click.ClickException, match="No prompt"):
            _get_prompt_or_stdin(None)


# ---------------------------------------------------------------------------
# 7. chat command
# ---------------------------------------------------------------------------

def test_chat_non_streaming(runner, valid_config):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "Hello!"}}],
    }

    with (
        patch("model_gateway.cli._try_worker_chat", return_value=False),
        patch("model_gateway.cli.load_config", return_value=valid_config),
        patch("model_gateway.cli._resolve_api_base", return_value=("http://localhost:8801/v1", "qwen3")),
        patch("model_gateway.cli.httpx.post", return_value=mock_resp),
    ):
        result = runner.invoke(cli, ["chat", "--no-stream", "Say hello"])

    assert result.exit_code == 0
    assert "Hello!" in result.output


def test_chat_with_system_prompt(runner, valid_config):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "4"}}],
    }

    with (
        patch("model_gateway.cli._try_worker_chat", return_value=False),
        patch("model_gateway.cli.load_config", return_value=valid_config),
        patch("model_gateway.cli._resolve_api_base", return_value=("http://localhost:8801/v1", "qwen3")),
        patch("model_gateway.cli.httpx.post", return_value=mock_resp) as mock_post,
    ):
        result = runner.invoke(cli, ["chat", "--no-stream", "-s", "Be terse", "What is 2+2?"])

    assert result.exit_code == 0
    payload = mock_post.call_args[1]["json"]
    assert payload["messages"][0] == {"role": "system", "content": "Be terse"}
    assert payload["messages"][1] == {"role": "user", "content": "What is 2+2?"}


def test_chat_unknown_model(runner, valid_config):
    with (
        patch("model_gateway.cli._try_worker_chat", return_value=False),
        patch("model_gateway.cli.load_config", return_value=valid_config),
    ):
        result = runner.invoke(cli, ["chat", "--no-stream", "-m", "nonexistent", "hi"])

    assert result.exit_code != 0
    assert "Unknown model" in result.output


def test_chat_backend_unreachable(runner, valid_config):
    import httpx

    with (
        patch("model_gateway.cli._try_worker_chat", return_value=False),
        patch("model_gateway.cli.load_config", return_value=valid_config),
        patch("model_gateway.cli._resolve_api_base", return_value=("http://localhost:8801/v1", "qwen3")),
        patch("model_gateway.cli.httpx.post", side_effect=httpx.ConnectError("refused")),
    ):
        result = runner.invoke(cli, ["chat", "--no-stream", "hi"])

    assert result.exit_code != 0
    assert "Cannot connect" in result.output


# ---------------------------------------------------------------------------
# 8. embed command
# ---------------------------------------------------------------------------

def test_embed_text_arg(runner, valid_config):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "data": [{"embedding": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}],
    }

    with (
        patch("model_gateway.cli._try_worker_embed", return_value=False),
        patch("model_gateway.cli.load_config", return_value=valid_config),
        patch("model_gateway.cli._resolve_api_base", return_value=("http://localhost:8801/v1", "qwen3")),
        patch("model_gateway.cli.httpx.post", return_value=mock_resp),
    ):
        result = runner.invoke(cli, ["embed", "Hello world"])

    assert result.exit_code == 0
    assert "Dimensions: 6" in result.output


def test_embed_from_file(runner, valid_config, tmp_path):
    input_file = tmp_path / "doc.txt"
    input_file.write_text("Some text to embed")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "data": [{"embedding": [0.1, 0.2]}],
    }

    with (
        patch("model_gateway.cli._try_worker_embed", return_value=False),
        patch("model_gateway.cli.load_config", return_value=valid_config),
        patch("model_gateway.cli._resolve_api_base", return_value=("http://localhost:8801/v1", "qwen3")),
        patch("model_gateway.cli.httpx.post", return_value=mock_resp) as mock_post,
    ):
        result = runner.invoke(cli, ["embed", "--file", str(input_file)])

    assert result.exit_code == 0
    payload = mock_post.call_args[1]["json"]
    assert payload["input"] == "Some text to embed"


# ---------------------------------------------------------------------------
# 9. complete command
# ---------------------------------------------------------------------------

def test_complete_text(runner, valid_config):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"text": " there lived a dragon."}],
    }

    with (
        patch("model_gateway.cli._try_worker_complete", return_value=False),
        patch("model_gateway.cli.load_config", return_value=valid_config),
        patch("model_gateway.cli._resolve_api_base", return_value=("http://localhost:8801/v1", "qwen3")),
        patch("model_gateway.cli.httpx.post", return_value=mock_resp),
    ):
        result = runner.invoke(cli, ["complete", "Once upon a time"])

    assert result.exit_code == 0
    assert "there lived a dragon" in result.output


def test_complete_with_max_tokens(runner, valid_config):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"text": "done"}],
    }

    with (
        patch("model_gateway.cli._try_worker_complete", return_value=False),
        patch("model_gateway.cli.load_config", return_value=valid_config),
        patch("model_gateway.cli._resolve_api_base", return_value=("http://localhost:8801/v1", "qwen3")),
        patch("model_gateway.cli.httpx.post", return_value=mock_resp) as mock_post,
    ):
        result = runner.invoke(cli, ["complete", "--max-tokens", "50", "Hello"])

    assert result.exit_code == 0
    payload = mock_post.call_args[1]["json"]
    assert payload["max_tokens"] == 50
