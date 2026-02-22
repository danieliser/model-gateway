"""Tests for the CLI worker (Unix socket daemon)."""

import asyncio
import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from model_gateway.config import GatewayConfig, ModelConfig, BackendConfig
from model_gateway.worker import Worker


@pytest.fixture
def valid_config():
    return GatewayConfig(
        port=8800,
        default_model="qwen3",
        models={
            "qwen3": ModelConfig(backend="mlx", model_id="mlx-community/Qwen3-4B-4bit"),
            "haiku": ModelConfig(backend="anthropic", model_id="claude-haiku-4-5-20251001"),
        },
        backends={
            "mlx": BackendConfig(enabled=True),
            "anthropic": BackendConfig(enabled=True),
        },
    )


# ---------------------------------------------------------------------------
# Resolution tests
# ---------------------------------------------------------------------------

def test_resolve_local_model(valid_config):
    with patch("model_gateway.worker.load_config", return_value=valid_config):
        w = Worker()
    url, model_id = w._resolve("qwen3")
    assert url == "http://localhost:8801/v1"
    assert model_id == "mlx-community/Qwen3-4B-4bit"


def test_resolve_cloud_model(valid_config):
    with patch("model_gateway.worker.load_config", return_value=valid_config):
        w = Worker()
    url, model_id = w._resolve("haiku")
    assert url == "http://localhost:8800/v1"
    assert model_id == "haiku"


def test_resolve_unknown_model(valid_config):
    with patch("model_gateway.worker.load_config", return_value=valid_config):
        w = Worker()
    with pytest.raises(ValueError, match="Unknown model"):
        w._resolve("nonexistent")


# ---------------------------------------------------------------------------
# Socket client tests
# ---------------------------------------------------------------------------

def test_socket_client_send_receive():
    from model_gateway.socket_client import send_request, iter_responses
    import socket

    # Create a socketpair to simulate client/server
    server_sock, client_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)

    # Client sends request
    send_request(client_sock, {"type": "ping"})

    # Server reads it
    data = server_sock.recv(8192)
    req = json.loads(data.decode().strip())
    assert req["type"] == "ping"

    # Server sends response
    server_sock.sendall((json.dumps({"pong": True}) + "\n").encode())
    server_sock.close()

    # Client reads response
    responses = list(iter_responses(client_sock))
    assert len(responses) == 1
    assert responses[0]["pong"] is True

    client_sock.close()


# ---------------------------------------------------------------------------
# Handler tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_ping(valid_config):
    with patch("model_gateway.worker.load_config", return_value=valid_config):
        w = Worker()

    reader = asyncio.StreamReader()
    reader.feed_data(json.dumps({"type": "ping"}).encode() + b"\n")
    reader.feed_eof()

    writer = MagicMock()
    written = []
    writer.write = lambda data: written.append(data)
    writer.drain = AsyncMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()

    await w.handle_connection(reader, writer)

    assert len(written) == 1
    resp = json.loads(written[0].decode().strip())
    assert resp["pong"] is True
    assert "uptime_s" in resp


@pytest.mark.asyncio
async def test_handle_unknown_type(valid_config):
    with patch("model_gateway.worker.load_config", return_value=valid_config):
        w = Worker()

    reader = asyncio.StreamReader()
    reader.feed_data(json.dumps({"type": "unknown_thing"}).encode() + b"\n")
    reader.feed_eof()

    writer = MagicMock()
    written = []
    writer.write = lambda data: written.append(data)
    writer.drain = AsyncMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()

    await w.handle_connection(reader, writer)

    assert len(written) == 1
    resp = json.loads(written[0].decode().strip())
    assert "error" in resp
    assert "unknown_thing" in resp["error"].lower()
