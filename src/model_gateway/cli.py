"""CLI entry point for model-gateway."""

import os
import signal
import subprocess
import sys
import time

import click
import httpx

from model_gateway.config import (
    get_config_dir,
    get_log_dir,
    get_pid_file,
    load_config,
    validate_config,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_process_running(pid: int) -> bool:
    """Check if a process with the given PID exists."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we can't signal it
        return True


def _find_pid_on_port(port: int) -> int | None:
    """Find the PID of the process listening on the given port."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            # May return multiple PIDs (one per line); take the first
            return int(result.stdout.strip().splitlines()[0])
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        pass
    return None


def _kill_pid(pid: int) -> None:
    """SIGTERM a process, wait, SIGKILL if needed."""
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(20):
        time.sleep(0.5)
        if not _is_process_running(pid):
            return
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _wait_for_health(port: int, timeout: float = 15.0) -> bool:
    """Poll gateway /health endpoint until it responds or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = httpx.get(f"http://localhost:{port}/health", timeout=1.0)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def _gateway_request(method: str, path: str, port: int = 8800, **kwargs):
    """Make HTTP request to running gateway. Returns None if gateway not running."""
    url = f"http://localhost:{port}{path}"
    try:
        resp = httpx.request(method, url, timeout=5.0, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        return None
    except Exception:
        return None


def _resolve_api_base(config, model_alias: str) -> tuple[str, str]:
    """Resolve model alias to (api_base, model_to_send).

    Local backends → direct to backend port (skip gateway, minimal hops).
    Cloud backends → through gateway (handles auth + LiteLLM routing).
    """
    model_cfg = config.models.get(model_alias)
    if model_cfg is None:
        available = sorted(config.models.keys())
        raise click.ClickException(
            f"Unknown model '{model_alias}'. Available: {', '.join(available)}"
        )

    # All backends go through the gateway (in-process backends have no port)
    return f"http://localhost:{config.port}/v1", model_alias


def _get_prompt_or_stdin(prompt: str | None) -> str:
    """Get prompt from argument or stdin."""
    if prompt:
        return prompt
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    raise click.ClickException("No prompt provided. Pass as argument or pipe via stdin.")


def _default_model(config) -> str:
    """Return the default model alias from config."""
    if config.default_model:
        return config.default_model
    if config.models:
        return next(iter(config.models))
    raise click.ClickException("No models configured.")


# ---------------------------------------------------------------------------
# Main group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option()
@click.option("--quiet", "-q", is_flag=True, help="Suppress informational output.")
@click.pass_context
def cli(ctx, quiet):
    """Model Gateway — manage local AI model routing and backends."""
    ctx.ensure_object(dict)
    ctx.obj["quiet"] = quiet


# ---------------------------------------------------------------------------
# Query commands
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("prompt", required=False)
@click.option("--model", "-m", default=None, help="Model alias from config.")
@click.option("--system", "-s", default=None, help="System prompt.")
@click.option("--max-tokens", default=None, type=int, help="Max tokens to generate.")
@click.option("--stream/--no-stream", default=True, help="Stream output (default: on).")
@click.option("--json", "json_output", is_flag=True, help="Output raw JSON response.")
def chat(prompt, model, system, max_tokens, stream, json_output):
    """Send a chat completion (tries worker, then direct, then gateway).

    \b
    Examples:
      model-gateway chat "Say hello" -m qwen3-4b
      model-gateway chat --system "Be terse" "What is 2+2?"
      echo "Explain gravity" | model-gateway chat -m qwen3-4b
    """
    config = _load_config_or_exit()
    model = model or _default_model(config)
    prompt_text = _get_prompt_or_stdin(prompt)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt_text})

    # Try 1: persistent worker (fastest — warm pool, no startup)
    if _try_worker_chat(model, messages, stream, max_tokens, json_output):
        return

    # Try 2: direct httpx (local) or gateway (cloud)
    api_base, model_id = _resolve_api_base(config, model)
    payload = {
        "model": model_id,
        "messages": messages,
        "stream": stream,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    url = f"{api_base}/chat/completions"
    start_ts = time.perf_counter()

    try:
        if stream and not json_output:
            _stream_chat(url, payload)
        else:
            resp = httpx.post(url, json=payload, timeout=120.0)
            resp.raise_for_status()
            data = resp.json()
            if json_output:
                import json as _json
                click.echo(_json.dumps(data, indent=2))
            else:
                content = data["choices"][0]["message"]["content"]
                click.echo(content)
    except httpx.ConnectError:
        raise click.ClickException("Cannot connect. Is the backend or gateway running?")
    except Exception as e:
        raise click.ClickException(f"Request failed: {e}")

    latency = (time.perf_counter() - start_ts) * 1000
    click.echo(f"\n({latency:.0f}ms)", err=True)


def _stream_chat(url: str, payload: dict) -> None:
    """Stream SSE chat response, printing tokens as they arrive."""
    import httpx
    import json as _json

    with httpx.stream("POST", url, json=payload, timeout=120.0) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break
            try:
                chunk = _json.loads(data_str)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    click.echo(content, nl=False)
            except _json.JSONDecodeError:
                pass
    click.echo()  # final newline


@cli.command()
@click.argument("text", required=False)
@click.option("--model", "-m", default=None, help="Model alias from config.")
@click.option("--file", "-f", "input_file", type=click.Path(exists=True), help="Read text from file.")
def embed(text, model, input_file):
    """Generate embeddings (tries worker, then direct, then gateway).

    \b
    Examples:
      model-gateway embed "Hello world" -m qwen3-4b
      model-gateway embed --file document.txt -m qwen3-4b
    """
    import json as _json

    config = _load_config_or_exit()
    model = model or config.embedding_model or _default_model(config)

    if input_file:
        with open(input_file) as f:
            text = f.read().strip()
    elif text is None:
        if not sys.stdin.isatty():
            text = sys.stdin.read().strip()
        else:
            raise click.ClickException("No text provided. Pass as argument, --file, or pipe via stdin.")

    # Try 1: persistent worker
    if _try_worker_embed(model, text):
        return

    # Try 2: direct httpx
    api_base, model_id = _resolve_api_base(config, model)
    url = f"{api_base}/embeddings"

    start_ts = time.perf_counter()
    try:
        resp = httpx.post(url, json={"model": model_id, "input": text}, timeout=120.0)
        resp.raise_for_status()
        data = resp.json()
    except httpx.ConnectError:
        raise click.ClickException("Cannot connect. Is the backend or gateway running?")
    except Exception as e:
        raise click.ClickException(f"Request failed: {e}")

    latency = (time.perf_counter() - start_ts) * 1000

    embeddings = data.get("data", [])
    if embeddings:
        vec = embeddings[0].get("embedding", [])
        click.echo(f"Dimensions: {len(vec)}")
        preview = ", ".join(f"{v:.6f}" for v in vec[:5])
        click.echo(f"Preview:    [{preview}, ...]")
        click.echo(_json.dumps(data), err=True)
    else:
        click.echo(_json.dumps(data, indent=2))

    click.echo(f"({latency:.0f}ms)", err=True)


@cli.command()
@click.argument("prompt", required=False)
@click.option("--model", "-m", default=None, help="Model alias from config.")
@click.option("--max-tokens", default=100, type=int, help="Max tokens to generate.")
def complete(prompt, model, max_tokens):
    """Send a text completion (tries worker, then direct, then gateway).

    \b
    Examples:
      model-gateway complete "Once upon a time" -m qwen3-4b
      model-gateway complete "The meaning of life is" --max-tokens 50
    """
    config = _load_config_or_exit()
    model = model or _default_model(config)
    prompt_text = _get_prompt_or_stdin(prompt)

    # Try 1: persistent worker
    if _try_worker_complete(model, prompt_text, max_tokens):
        return

    # Try 2: direct httpx
    api_base, model_id = _resolve_api_base(config, model)
    url = f"{api_base}/completions"
    payload = {
        "model": model_id,
        "prompt": prompt_text,
        "max_tokens": max_tokens,
    }

    start_ts = time.perf_counter()
    try:
        resp = httpx.post(url, json=payload, timeout=120.0)
        resp.raise_for_status()
        data = resp.json()
    except httpx.ConnectError:
        raise click.ClickException("Cannot connect. Is the backend or gateway running?")
    except Exception as e:
        raise click.ClickException(f"Request failed: {e}")

    latency = (time.perf_counter() - start_ts) * 1000

    choices = data.get("choices", [])
    if choices:
        click.echo(choices[0].get("text", ""))
    else:
        import json as _json
        click.echo(_json.dumps(data, indent=2))

    click.echo(f"\n({latency:.0f}ms)", err=True)


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--port", default=None, type=int, help="Override config port.")
@click.option("--config", "config_path", default=None, help="Path to config file.")
@click.option("--foreground", "-f", is_flag=True, help="Run in foreground (don't daemonize).")
@click.pass_context
def start(ctx, port, config_path, foreground):
    """Start the gateway server."""
    quiet = ctx.obj.get("quiet", False)

    try:
        config = load_config(config_path)
    except FileNotFoundError as e:
        click.secho(str(e), fg="red", err=True)
        raise SystemExit(1)

    if port:
        config.port = port

    errors = validate_config(config)
    for msg in errors:
        click.secho(f"Warning: {msg}", fg="yellow", err=True)

    pid_file = get_pid_file()
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            if _is_process_running(pid):
                if not quiet:
                    click.echo(f"Gateway already running (PID {pid})")
                return
        except (ValueError, OSError):
            pass
        pid_file.unlink(missing_ok=True)

    if foreground:
        import uvicorn
        uvicorn.run(
            "model_gateway.server:app",
            host="0.0.0.0",
            port=config.port,
        )
    else:
        log_file = get_log_dir() / "gateway.log"
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn", "model_gateway.server:app",
                "--host", "0.0.0.0",
                "--port", str(config.port),
            ],
            stdout=open(log_file, "a"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        pid_file.write_text(str(proc.pid))
        if _wait_for_health(config.port):
            if not quiet:
                click.secho(
                    f"Gateway started on port {config.port} (PID {proc.pid})",
                    fg="green",
                )
        else:
            click.secho("Gateway failed to start. Check logs.", fg="red", err=True)
            pid_file.unlink(missing_ok=True)
            raise SystemExit(1)


@cli.command()
@click.pass_context
def stop(ctx):
    """Stop the running gateway server."""
    quiet = ctx.obj.get("quiet", False)

    try:
        config = load_config()
        port = config.port
    except FileNotFoundError:
        port = 8800

    pids_to_kill: set[int] = set()

    # 1. PID from file
    pid_file = get_pid_file()
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            if _is_process_running(pid):
                pids_to_kill.add(pid)
        except (ValueError, OSError):
            pass

    # 2. Find process actually listening on the port (handles stale PID file)
    port_pid = _find_pid_on_port(port)
    if port_pid:
        pids_to_kill.add(port_pid)

    if not pids_to_kill:
        pid_file.unlink(missing_ok=True)
        if not quiet:
            click.echo("Gateway is not running.")
        return

    for pid in pids_to_kill:
        _kill_pid(pid)

    pid_file.unlink(missing_ok=True)
    if not quiet:
        click.secho("Gateway stopped.", fg="green")


@cli.command()
@click.option("--port", default=None, type=int, help="Override config port.")
@click.option("--config", "config_path", default=None, help="Path to config file.")
@click.pass_context
def restart(ctx, port, config_path):
    """Restart the gateway server."""
    ctx.invoke(stop)
    ctx.invoke(start, port=port, config_path=config_path)


# ---------------------------------------------------------------------------
# Status and info
# ---------------------------------------------------------------------------

@cli.command()
@click.pass_context
def status(ctx):
    """Show gateway server status."""
    pid_file = get_pid_file()

    if not pid_file.exists():
        click.echo("Gateway:    not running")
        click.echo("\nRun 'model-gateway start' to start the gateway.")
        return

    try:
        pid = int(pid_file.read_text().strip())
    except (ValueError, OSError):
        click.echo("Gateway:    not running (stale PID file)")
        return

    if not _is_process_running(pid):
        click.echo("Gateway:    not running (stale PID file)")
        pid_file.unlink(missing_ok=True)
        return

    try:
        config = load_config()
    except FileNotFoundError:
        config = None

    port = config.port if config else 8800
    data = _gateway_request("GET", "/health", port=port)

    if data is None:
        click.echo(f"Gateway:    running (PID {pid}) — not responding to /health")
        return

    click.echo(f"Gateway:    running (PID {pid})")
    click.echo(f"Port:       {port}")

    uptime = data.get("uptime_seconds")
    if uptime is not None:
        h = int(uptime // 3600)
        m = int((uptime % 3600) // 60)
        click.echo(f"Uptime:     {h}h {m}m")

    active_backend = data.get("active_backend")
    active_model = data.get("active_model")
    if active_backend:
        click.echo(f"\nActive local backend: {active_backend}")
    if active_model:
        click.echo(f"Active local model:   {active_model}")

    backends = data.get("backends", {})
    if backends:
        click.echo("\nAvailable backends:")
        for name, info in backends.items():
            ok = info.get("available", False)
            mark = click.style("✓", fg="green") if ok else click.style("✗", fg="red")
            reason = info.get("reason", "")
            detail = f" ({reason})" if reason else ""
            click.echo(f"  {mark} {name:<12}{detail}")


@cli.command()
@click.pass_context
def models(ctx):
    """List available models and their backend status."""
    try:
        config = load_config()
    except FileNotFoundError as e:
        click.secho(str(e), fg="red", err=True)
        raise SystemExit(1)

    data = _gateway_request("GET", "/v1/models", port=config.port)

    if data is None:
        # Fall back to static config display
        click.echo(f"{'ALIAS':<15} {'BACKEND':<12} {'MODEL':<40} STATUS")
        for alias, m in config.models.items():
            model_id = m.model_id or m.model_path or "(not set)"
            click.echo(f"{alias:<15} {m.backend:<12} {model_id:<40} (gateway offline)")
        return

    entries = data.get("data", data) if isinstance(data, dict) else data
    click.echo(f"{'ALIAS':<15} {'BACKEND':<12} {'MODEL':<40} STATUS")
    for entry in entries:
        alias = entry.get("alias", "")
        backend = entry.get("backend", "")
        model_id = entry.get("model_id", entry.get("model", ""))
        loaded = entry.get("loaded", False)
        status_str = click.style("● loaded", fg="green") if loaded else click.style("○ idle", fg="yellow")
        click.echo(f"{alias:<15} {backend:<12} {model_id:<40} {status_str}")


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("alias")
@click.option("--type", "-t", "model_type", default=None,
              type=click.Choice(["chat", "embed", "rerank", "tts"]),
              help="Set as default for a specific type (chat, embed, rerank, tts).")
@click.pass_context
def switch(ctx, alias, model_type):
    """Switch the default model to ALIAS.

    \b
    Without --type, sets the global default_model.
    With --type, sets the type-specific default:
      gateway switch kokoro --type tts
      gateway switch nomic-embed --type embed
      gateway switch qwen3-8b --type chat
    """
    try:
        config = load_config()
    except FileNotFoundError as e:
        click.secho(str(e), fg="red", err=True)
        raise SystemExit(1)

    payload = {"model": alias}
    if model_type:
        payload["model_type"] = model_type

    result = _gateway_request("POST", "/gateway/switch", port=config.port, json=payload)
    if result is None:
        click.secho(
            "Gateway not running. Run 'model-gateway start' first.", fg="red", err=True
        )
        raise SystemExit(1)

    label = model_type or "default"
    click.secho(f"Switched {label} model to: {alias}", fg="green")


@cli.command()
@click.argument("alias")
@click.pass_context
def test(ctx, alias):
    """Send a test prompt through the gateway server."""
    try:
        config = load_config()
    except FileNotFoundError as e:
        click.secho(str(e), fg="red", err=True)
        raise SystemExit(1)


    prompt = "Say 'hello' in one word."
    payload = {
        "model": alias,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 20,
    }

    url = f"http://localhost:{config.port}/v1/chat/completions"
    start_ts = time.monotonic()
    try:
        resp = httpx.post(url, json=payload, timeout=30.0)
        latency = time.monotonic() - start_ts
        resp.raise_for_status()
        data = resp.json()
    except httpx.ConnectError:
        click.secho(
            "Gateway not running. Run 'model-gateway start' first.", fg="red", err=True
        )
        raise SystemExit(1)
    except Exception as e:
        click.secho(f"Request failed: {e}", fg="red", err=True)
        raise SystemExit(1)

    content = data["choices"][0]["message"]["content"]
    click.echo(f"Response:  {content}")
    click.echo(f"Latency:   {latency * 1000:.0f}ms")


# ---------------------------------------------------------------------------
# Config management
# ---------------------------------------------------------------------------

@cli.group()
def config():
    """Manage gateway configuration."""


@config.command("show")
def config_show():
    """Print the current resolved configuration."""
    try:
        cfg = load_config()
    except FileNotFoundError as e:
        click.secho(str(e), fg="red", err=True)
        raise SystemExit(1)

    import dataclasses
    import json

    def _redact(obj):
        if dataclasses.is_dataclass(obj):
            d = {}
            for f in dataclasses.fields(obj):
                val = getattr(obj, f.name)
                if "key" in f.name.lower() or "secret" in f.name.lower():
                    d[f.name] = "***" if val else None
                else:
                    d[f.name] = _redact(val)
            return d
        if isinstance(obj, dict):
            return {k: _redact(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_redact(v) for v in obj]
        return obj

    click.echo(json.dumps(_redact(cfg), indent=2))


@config.command("edit")
def config_edit():
    """Open the config file in $EDITOR."""
    config_dir = get_config_dir()
    config_file = config_dir / "config.yml"

    if not config_file.exists():
        click.echo(f"Creating new config at {config_file}")
        config_file.write_text(
            "# Model Gateway Configuration\n"
            "port: 8800\n"
            "# default_model: my-model\n"
            "models: {}\n"
            "backends: {}\n"
        )

    editor = os.environ.get("EDITOR", "vi")
    os.execvp(editor, [editor, str(config_file)])


@config.command("validate")
def config_validate():
    """Validate the config file for errors."""
    try:
        cfg = load_config()
    except FileNotFoundError as e:
        click.secho(str(e), fg="red", err=True)
        raise SystemExit(1)

    errors = validate_config(cfg)
    if not errors:
        click.secho("Config is valid.", fg="green")
    else:
        for msg in errors:
            click.secho(f"  Error: {msg}", fg="red")
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--follow", "-f", is_flag=True, help="Follow log output.")
@click.option("--lines", "-n", default=50, help="Number of lines to show.")
def logs(follow, lines):
    """Show gateway server logs."""
    log_file = get_log_dir() / "gateway.log"

    if not log_file.exists():
        click.echo(f"No log file found at {log_file}")
        return

    if follow:
        import subprocess as sp
        try:
            sp.run(["tail", f"-n{lines}", "-f", str(log_file)])
        except KeyboardInterrupt:
            pass
    else:
        import subprocess as sp
        sp.run(["tail", f"-n{lines}", str(log_file)])


@cli.command()
@click.pass_context
def health(ctx):
    """Check health of the gateway and all configured backends."""
    try:
        config = load_config()
    except FileNotFoundError as e:
        click.secho(str(e), fg="red", err=True)
        raise SystemExit(1)

    data = _gateway_request("GET", "/health", port=config.port)
    if data is None:
        click.secho(
            "Gateway not running. Run 'model-gateway start' first.", fg="red", err=True
        )
        raise SystemExit(1)

    overall = data.get("status", "unknown")
    color = "green" if overall in ("ok", "healthy") else "yellow" if overall == "degraded" else "red"
    click.secho(f"Status:  {overall}", fg=color)

    backends = data.get("backends", {})
    if backends:
        click.echo("\nBackends:")
        for name, info in backends.items():
            ok = info.get("running", False)
            mark = click.style("✓", fg="green") if ok else click.style("✗", fg="red")
            model = info.get("model", "")
            detail = f" ({model})" if model and ok else ""
            click.echo(f"  {mark} {name}{detail}")


@cli.command()
@click.argument("text")
@click.option("--model", "-m", default=None, help="TTS model alias (default: first TTS model in config)")
@click.option("--voice", default=None, help="Voice ID (default: from config)")
@click.option("--speed", type=float, default=None, help="Speed multiplier (default: 1.0)")
@click.option("--output", "-o", default=None, help="Save to file instead of playing")
@click.pass_context
def speak(ctx, text, model, voice, speed, output):
    """Generate and play TTS audio."""
    config = _load_config_or_exit()

    payload = {"input": text, "response_format": "wav"}
    if model:
        payload["model"] = model
    if voice:
        payload["voice"] = voice
    if speed:
        payload["speed"] = speed

    url = f"http://localhost:{config.port}/v1/audio/speech"
    try:
        resp = httpx.post(url, json=payload, timeout=60.0)
        resp.raise_for_status()
        data = resp.content
    except httpx.ConnectError:
        raise click.ClickException("Gateway not running. Start with: model-gateway start")
    except httpx.HTTPStatusError as e:
        raise click.ClickException(f"TTS request failed: {e.response.text}")

    if output:
        with open(output, "wb") as f:
            f.write(data)
        click.echo(f"Saved to {output}")
    else:
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="tts-cli-")
        tmp.write(data)
        tmp.close()
        try:
            import subprocess as sp
            sp.run(["play", "-q", tmp.name], check=False, timeout=30)
        except FileNotFoundError:
            try:
                sp.run(["afplay", tmp.name], check=False, timeout=30)
            except FileNotFoundError:
                click.echo(f"Audio saved to {tmp.name} (no player found)")
                return
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Worker lifecycle
# ---------------------------------------------------------------------------

@cli.group()
def worker():
    """Manage the persistent CLI worker (fast socket-based requests)."""


@worker.command("start")
@click.pass_context
def worker_start(ctx):
    """Start the CLI worker daemon."""
    from model_gateway.worker import get_socket_path, get_worker_pid_file

    pid_file = get_worker_pid_file()
    if os.path.exists(pid_file):
        try:
            pid = int(open(pid_file).read().strip())
            if _is_process_running(pid):
                click.echo(f"Worker already running (PID {pid})")
                return
        except (ValueError, OSError):
            pass

    _start_worker_daemon()
    quiet = ctx.obj.get("quiet", False)
    if not quiet:
        socket_path = get_socket_path()
        click.secho(f"Worker started, listening on {socket_path}", fg="green")


@worker.command("stop")
@click.pass_context
def worker_stop(ctx):
    """Stop the CLI worker daemon."""
    from model_gateway.worker import get_worker_pid_file

    pid_file = get_worker_pid_file()
    if not os.path.exists(pid_file):
        click.echo("Worker is not running.")
        return

    try:
        pid = int(open(pid_file).read().strip())
    except (ValueError, OSError):
        os.unlink(pid_file)
        click.echo("Worker is not running.")
        return

    try:
        os.kill(pid, signal.SIGTERM)
        for _ in range(20):
            time.sleep(0.25)
            if not _is_process_running(pid):
                break
    except ProcessLookupError:
        pass

    if os.path.exists(pid_file):
        os.unlink(pid_file)
    quiet = ctx.obj.get("quiet", False)
    if not quiet:
        click.secho("Worker stopped.", fg="green")


@worker.command("status")
def worker_status():
    """Show CLI worker status."""
    from model_gateway.worker import get_socket_path, get_worker_pid_file

    pid_file = get_worker_pid_file()
    socket_path = get_socket_path()

    if not os.path.exists(pid_file):
        click.echo("Worker:  not running")
        return

    try:
        pid = int(open(pid_file).read().strip())
    except (ValueError, OSError):
        click.echo("Worker:  not running (stale PID file)")
        return

    if not _is_process_running(pid):
        click.echo("Worker:  not running (stale PID file)")
        return

    click.echo(f"Worker:  running (PID {pid})")
    click.echo(f"Socket:  {socket_path}")

    # Ping for uptime
    resp = _worker_request({"type": "ping"})
    if resp and resp.get("pong"):
        uptime = resp.get("uptime_s", 0)
        m, s = divmod(uptime, 60)
        h, m = divmod(m, 60)
        click.echo(f"Uptime:  {int(h)}h {int(m)}m")


# ---------------------------------------------------------------------------
# Worker helpers
# ---------------------------------------------------------------------------

def _worker_request(request: dict) -> dict | None:
    """Send a request to the worker, return first response or None if unavailable."""
    from model_gateway.worker import get_socket_path
    from model_gateway.socket_client import connect, send_request, iter_responses

    socket_path = get_socket_path()
    if not os.path.exists(socket_path):
        return None

    try:
        sock = connect(socket_path)
        send_request(sock, request)
        for resp in iter_responses(sock):
            return resp
    except (ConnectionRefusedError, OSError):
        return None
    return None


def _start_worker_daemon() -> None:
    """Fork the worker process in the background."""
    from model_gateway.worker import get_socket_path

    proc = subprocess.Popen(
        [sys.executable, "-m", "model_gateway.worker"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    # Wait for socket to appear
    socket_path = get_socket_path()
    for _ in range(40):
        if os.path.exists(socket_path):
            return
        time.sleep(0.05)


def _try_worker_chat(model, messages, stream, max_tokens, json_output) -> bool:
    """Try sending a chat request through the worker. Returns True if handled."""
    from model_gateway.worker import get_socket_path
    from model_gateway.socket_client import connect, send_request, iter_responses

    socket_path = get_socket_path()
    if not os.path.exists(socket_path):
        return False

    request = {
        "type": "chat",
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    if max_tokens is not None:
        request["max_tokens"] = max_tokens

    try:
        sock = connect(socket_path)
        send_request(sock, request)
        for resp in iter_responses(sock):
            if "error" in resp:
                raise click.ClickException(resp["error"])
            if "chunk" in resp:
                click.echo(resp["chunk"], nl=False)
            elif "content" in resp:
                if json_output:
                    click.echo(resp["content"])
                else:
                    click.echo(resp["content"])
            if resp.get("done"):
                if stream and "chunk" not in resp:
                    pass  # no trailing newline needed
                elif stream:
                    click.echo()  # final newline after streamed chunks
                latency = resp.get("latency_ms", 0)
                click.echo(f"\n({latency}ms)", err=True)
                break
        sock.close()
        return True
    except (ConnectionRefusedError, OSError):
        return False


def _try_worker_embed(model, text) -> bool:
    """Try sending an embed request through the worker. Returns True if handled."""
    import json as _json
    from model_gateway.worker import get_socket_path
    from model_gateway.socket_client import connect, send_request, iter_responses

    socket_path = get_socket_path()
    if not os.path.exists(socket_path):
        return False

    try:
        sock = connect(socket_path)
        send_request(sock, {"type": "embed", "model": model, "text": text})
        for resp in iter_responses(sock):
            if "error" in resp:
                raise click.ClickException(resp["error"])
            if resp.get("done"):
                data = resp.get("data", {})
                embeddings = data.get("data", [])
                if embeddings:
                    vec = embeddings[0].get("embedding", [])
                    click.echo(f"Dimensions: {len(vec)}")
                    preview = ", ".join(f"{v:.6f}" for v in vec[:5])
                    click.echo(f"Preview:    [{preview}, ...]")
                    click.echo(_json.dumps(data), err=True)
                latency = resp.get("latency_ms", 0)
                click.echo(f"({latency}ms)", err=True)
                break
        sock.close()
        return True
    except (ConnectionRefusedError, OSError):
        return False


def _try_worker_complete(model, prompt_text, max_tokens) -> bool:
    """Try sending a complete request through the worker. Returns True if handled."""
    from model_gateway.worker import get_socket_path
    from model_gateway.socket_client import connect, send_request, iter_responses

    socket_path = get_socket_path()
    if not os.path.exists(socket_path):
        return False

    try:
        sock = connect(socket_path)
        send_request(sock, {
            "type": "complete",
            "model": model,
            "prompt": prompt_text,
            "max_tokens": max_tokens,
        })
        for resp in iter_responses(sock):
            if "error" in resp:
                raise click.ClickException(resp["error"])
            if resp.get("done"):
                if resp.get("text"):
                    click.echo(resp["text"])
                latency = resp.get("latency_ms", 0)
                click.echo(f"\n({latency}ms)", err=True)
                break
        sock.close()
        return True
    except (ConnectionRefusedError, OSError):
        return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_config_or_exit():
    """Load config or exit with error message."""
    try:
        return load_config()
    except FileNotFoundError as e:
        raise click.ClickException(str(e))
