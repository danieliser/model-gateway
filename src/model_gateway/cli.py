"""CLI entry point for model-gateway."""

import os
import signal
import subprocess
import sys
import time

import click

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


def _wait_for_health(port: int, timeout: float = 15.0) -> bool:
    """Poll gateway /health endpoint until it responds or timeout."""
    import httpx

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
    import httpx

    url = f"http://localhost:{port}{path}"
    try:
        resp = httpx.request(method, url, timeout=5.0, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        return None
    except Exception:
        return None


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

    pid_file = get_pid_file()
    if not pid_file.exists():
        click.echo("Gateway is not running. Run 'gateway start' first.")
        return

    try:
        pid = int(pid_file.read_text().strip())
    except (ValueError, OSError):
        pid_file.unlink(missing_ok=True)
        click.echo("Gateway is not running.")
        return

    try:
        os.kill(pid, signal.SIGTERM)
        for _ in range(20):
            time.sleep(0.5)
            if not _is_process_running(pid):
                break
        else:
            os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass

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
        click.echo("\nRun 'gateway start' to start the gateway.")
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

    data = _gateway_request("GET", "/models", port=config.port)

    if data is None:
        # Fall back to static config display
        click.echo(f"{'ALIAS':<15} {'BACKEND':<12} {'MODEL':<40} STATUS")
        for alias, m in config.models.items():
            model_id = m.model_id or m.model_path or "(not set)"
            click.echo(f"{alias:<15} {m.backend:<12} {model_id:<40} (gateway offline)")
        return

    click.echo(f"{'ALIAS':<15} {'BACKEND':<12} {'MODEL':<40} STATUS")
    for entry in data:
        alias = entry.get("alias", "")
        backend = entry.get("backend", "")
        model_id = entry.get("model", "")
        ok = entry.get("available", False)
        status_str = click.style("✓ ready", fg="green") if ok else click.style("✗ unavailable", fg="red")
        click.echo(f"{alias:<15} {backend:<12} {model_id:<40} {status_str}")


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("alias")
@click.pass_context
def switch(ctx, alias):
    """Switch the default model to ALIAS."""
    try:
        config = load_config()
    except FileNotFoundError as e:
        click.secho(str(e), fg="red", err=True)
        raise SystemExit(1)

    result = _gateway_request("POST", "/gateway/switch", port=config.port, json={"model": alias})
    if result is None:
        click.secho(
            "Gateway not running. Run 'gateway start' first.", fg="red", err=True
        )
        raise SystemExit(1)

    click.secho(f"Switched to model: {alias}", fg="green")


@cli.command()
@click.argument("alias")
@click.pass_context
def test(ctx, alias):
    """Send a test prompt to model ALIAS."""
    try:
        config = load_config()
    except FileNotFoundError as e:
        click.secho(str(e), fg="red", err=True)
        raise SystemExit(1)

    import httpx

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
            "Gateway not running. Run 'gateway start' first.", fg="red", err=True
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
            "Gateway not running. Run 'gateway start' first.", fg="red", err=True
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
