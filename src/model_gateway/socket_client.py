"""Thin Unix socket client for the CLI worker.

Minimal imports (socket + json) to avoid adding startup overhead.
"""

import json
import socket


def connect(socket_path: str) -> socket.socket:
    """Connect to the worker Unix socket. Raises ConnectionRefusedError if not running."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(socket_path)
    return sock


def send_request(sock: socket.socket, request: dict) -> None:
    """Send a JSON request line to the worker."""
    sock.sendall((json.dumps(request) + "\n").encode())


def iter_responses(sock: socket.socket):
    """Yield parsed JSON response lines from the worker."""
    buf = b""
    while True:
        data = sock.recv(8192)
        if not data:
            break
        buf += data
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            if line:
                yield json.loads(line.decode())
