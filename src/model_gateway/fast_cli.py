"""Ultra-thin CLI client for the worker socket.

Bypasses click/httpx entirely — only uses stdlib (socket, json, sys).
Falls back to the full CLI if the worker isn't running.
"""

import json
import os
import socket
import sys
import time

SOCKET_PATH = os.path.expanduser("~/.config/model-gateway/cli.sock")


def main():
    args = sys.argv[1:]
    if not args:
        # No args — fall back to full CLI for --help etc.
        _fallback()
        return

    cmd = args[0]

    if cmd == "chat":
        _chat(args[1:])
    elif cmd == "embed":
        _embed(args[1:])
    elif cmd == "complete":
        _complete(args[1:])
    else:
        # Unknown command — fall back to full CLI
        _fallback()


def _chat(args):
    model = None
    system = None
    max_tokens = None
    stream = True
    prompt = None

    i = 0
    while i < len(args):
        a = args[i]
        if a in ("-m", "--model") and i + 1 < len(args):
            model = args[i + 1]; i += 2
        elif a in ("-s", "--system") and i + 1 < len(args):
            system = args[i + 1]; i += 2
        elif a == "--max-tokens" and i + 1 < len(args):
            max_tokens = int(args[i + 1]); i += 2
        elif a == "--no-stream":
            stream = False; i += 1
        elif a == "--stream":
            stream = True; i += 1
        elif not a.startswith("-"):
            prompt = a; i += 1
        else:
            i += 1

    if prompt is None and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    if prompt is None:
        print("Error: no prompt provided", file=sys.stderr)
        sys.exit(1)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    request = {"type": "chat", "model": model or "qwen3-4b", "messages": messages, "stream": stream}
    if max_tokens:
        request["max_tokens"] = max_tokens

    sock = _connect()
    if sock is None:
        _fallback()
        return

    _send(sock, request)
    for resp in _recv(sock):
        if "error" in resp:
            print(f"Error: {resp['error']}", file=sys.stderr)
            sys.exit(1)
        if "chunk" in resp:
            sys.stdout.write(resp["chunk"])
            sys.stdout.flush()
        elif "content" in resp:
            print(resp["content"])
        if resp.get("done"):
            if stream:
                print()  # newline after streamed chunks
            print(f"({resp.get('latency_ms', 0)}ms)", file=sys.stderr)
            break
    sock.close()


def _embed(args):
    model = None
    text = None

    i = 0
    while i < len(args):
        a = args[i]
        if a in ("-m", "--model") and i + 1 < len(args):
            model = args[i + 1]; i += 2
        elif a in ("-f", "--file") and i + 1 < len(args):
            with open(args[i + 1]) as f:
                text = f.read().strip()
            i += 2
        elif not a.startswith("-"):
            text = a; i += 1
        else:
            i += 1

    if text is None and not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    if text is None:
        print("Error: no text provided", file=sys.stderr)
        sys.exit(1)

    sock = _connect()
    if sock is None:
        _fallback()
        return

    _send(sock, {"type": "embed", "model": model or "qwen3-4b", "text": text})
    for resp in _recv(sock):
        if "error" in resp:
            print(f"Error: {resp['error']}", file=sys.stderr)
            sys.exit(1)
        if resp.get("done"):
            data = resp.get("data", {})
            embeddings = data.get("data", [])
            if embeddings:
                vec = embeddings[0].get("embedding", [])
                print(f"Dimensions: {len(vec)}")
                preview = ", ".join(f"{v:.6f}" for v in vec[:5])
                print(f"Preview:    [{preview}, ...]")
            print(f"({resp.get('latency_ms', 0)}ms)", file=sys.stderr)
            break
    sock.close()


def _complete(args):
    model = None
    max_tokens = 100
    prompt = None

    i = 0
    while i < len(args):
        a = args[i]
        if a in ("-m", "--model") and i + 1 < len(args):
            model = args[i + 1]; i += 2
        elif a == "--max-tokens" and i + 1 < len(args):
            max_tokens = int(args[i + 1]); i += 2
        elif not a.startswith("-"):
            prompt = a; i += 1
        else:
            i += 1

    if prompt is None and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    if prompt is None:
        print("Error: no prompt provided", file=sys.stderr)
        sys.exit(1)

    sock = _connect()
    if sock is None:
        _fallback()
        return

    _send(sock, {"type": "complete", "model": model or "qwen3-4b", "prompt": prompt, "max_tokens": max_tokens})
    for resp in _recv(sock):
        if "error" in resp:
            print(f"Error: {resp['error']}", file=sys.stderr)
            sys.exit(1)
        if resp.get("done"):
            if resp.get("text"):
                print(resp["text"])
            print(f"({resp.get('latency_ms', 0)}ms)", file=sys.stderr)
            break
    sock.close()


def _connect():
    if not os.path.exists(SOCKET_PATH):
        return None
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(SOCKET_PATH)
        return s
    except (ConnectionRefusedError, OSError):
        return None


def _send(sock, request):
    sock.sendall((json.dumps(request) + "\n").encode())


def _recv(sock):
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


def _fallback():
    """Exec the full CLI (click-based) for commands we don't handle."""
    # Try the installed entry point first, fall back to module invocation
    full_cli = os.path.join(os.path.dirname(sys.executable), "model-gateway-full")
    if os.path.exists(full_cli):
        os.execvp(full_cli, [full_cli] + sys.argv[1:])
    else:
        os.execvp(sys.executable, [sys.executable, "-m", "model_gateway.cli"] + sys.argv[1:])


if __name__ == "__main__":
    main()
