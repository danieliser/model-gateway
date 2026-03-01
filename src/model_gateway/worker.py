"""Persistent CLI worker — Unix socket listener with warm httpx connection pool.

Eliminates Python startup overhead for CLI commands by keeping a long-running
process that handles requests over a Unix domain socket.
"""

import asyncio
import json
import logging
import os
import signal
import time

import httpx

from model_gateway.config import CLOUD_BACKENDS, load_config

logger = logging.getLogger(__name__)


def get_socket_path() -> str:
    """Return the Unix socket path for the CLI worker."""
    config_dir = os.path.expanduser("~/.config/model-gateway")
    os.makedirs(config_dir, exist_ok=True)
    return os.path.join(config_dir, "cli.sock")


def get_worker_pid_file() -> str:
    config_dir = os.path.expanduser("~/.config/model-gateway")
    return os.path.join(config_dir, "worker.pid")


class Worker:
    def __init__(self):
        self._config = load_config()
        self._client = httpx.AsyncClient(timeout=120.0)
        self._start_time = time.monotonic()

    def _resolve(self, model_alias: str) -> tuple[str, str]:
        """Resolve model alias to (api_base, model_to_send)."""
        model_cfg = self._config.models.get(model_alias)
        if model_cfg is None:
            raise ValueError(f"Unknown model: {model_alias}")

        if model_cfg.backend in CLOUD_BACKENDS:
            return f"http://localhost:{self._config.port}/v1", model_alias

        model_id = model_cfg.model_id or model_alias
        return "http://localhost:8801/v1", model_id

    async def handle_chat(self, req: dict, writer: asyncio.StreamWriter) -> None:
        model = req["model"]
        api_base, model_id = self._resolve(model)
        url = f"{api_base}/chat/completions"

        payload = {
            "model": model_id,
            "messages": req["messages"],
            "stream": req.get("stream", True),
        }
        if req.get("max_tokens"):
            payload["max_tokens"] = req["max_tokens"]

        t0 = time.perf_counter()

        if payload["stream"]:
            async with self._client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            writer.write(
                                (json.dumps({"chunk": content}) + "\n").encode()
                            )
                            await writer.drain()
                    except json.JSONDecodeError:
                        pass
        else:
            resp = await self._client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            writer.write((json.dumps({"content": content}) + "\n").encode())
            await writer.drain()

        latency = (time.perf_counter() - t0) * 1000
        writer.write((json.dumps({"done": True, "latency_ms": round(latency)}) + "\n").encode())
        await writer.drain()

    async def handle_embed(self, req: dict, writer: asyncio.StreamWriter) -> None:
        model = req["model"]
        api_base, model_id = self._resolve(model)
        url = f"{api_base}/embeddings"

        t0 = time.perf_counter()
        resp = await self._client.post(
            url, json={"model": model_id, "input": req["text"]}
        )
        resp.raise_for_status()
        data = resp.json()
        latency = (time.perf_counter() - t0) * 1000

        writer.write(
            (json.dumps({"data": data, "done": True, "latency_ms": round(latency)}) + "\n").encode()
        )
        await writer.drain()

    async def handle_complete(self, req: dict, writer: asyncio.StreamWriter) -> None:
        model = req["model"]
        api_base, model_id = self._resolve(model)
        url = f"{api_base}/completions"

        payload = {
            "model": model_id,
            "prompt": req["prompt"],
            "max_tokens": req.get("max_tokens", 100),
        }

        t0 = time.perf_counter()
        resp = await self._client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        latency = (time.perf_counter() - t0) * 1000

        text = data.get("choices", [{}])[0].get("text", "")
        writer.write(
            (json.dumps({"text": text, "done": True, "latency_ms": round(latency)}) + "\n").encode()
        )
        await writer.drain()

    async def handle_tts(self, req: dict, writer: asyncio.StreamWriter) -> None:
        """Generate TTS audio via the gateway's /v1/audio/speech endpoint."""
        url = f"http://localhost:{self._config.port}/v1/audio/speech"

        payload = {
            "model": req.get("model"),
            "input": req["text"],
            "voice": req.get("voice"),
            "speed": req.get("speed", 1.0),
            "response_format": req.get("response_format", "wav"),
        }
        # Pass through optional params
        for key in ("lang_code", "exaggeration", "instruct", "conds_path", "ref_audio"):
            if key in req:
                payload[key] = req[key]

        t0 = time.perf_counter()
        resp = await self._client.post(url, json=payload)
        resp.raise_for_status()
        latency = (time.perf_counter() - t0) * 1000

        # Save to temp file and return the path
        import tempfile
        suffix = ".mp3" if req.get("response_format") == "mp3" else ".wav"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, prefix="tts-worker-")
        tmp.write(resp.content)
        tmp.close()

        writer.write(
            (json.dumps({"path": tmp.name, "done": True, "latency_ms": round(latency)}) + "\n").encode()
        )
        await writer.drain()

    async def handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            line = await reader.readline()
            if not line:
                return

            req = json.loads(line.decode())
            req_type = req.get("type")

            if req_type == "chat":
                await self.handle_chat(req, writer)
            elif req_type == "embed":
                await self.handle_embed(req, writer)
            elif req_type == "complete":
                await self.handle_complete(req, writer)
            elif req_type == "tts":
                await self.handle_tts(req, writer)
            elif req_type == "ping":
                uptime = time.monotonic() - self._start_time
                writer.write(
                    (json.dumps({"pong": True, "uptime_s": round(uptime)}) + "\n").encode()
                )
                await writer.drain()
            else:
                writer.write(
                    (json.dumps({"error": f"Unknown request type: {req_type}"}) + "\n").encode()
                )
                await writer.drain()

        except Exception as e:
            try:
                writer.write(
                    (json.dumps({"error": str(e), "done": True}) + "\n").encode()
                )
                await writer.drain()
            except Exception:
                pass
        finally:
            writer.close()
            await writer.wait_closed()

    async def run(self) -> None:
        socket_path = get_socket_path()

        # Clean up stale socket
        if os.path.exists(socket_path):
            os.unlink(socket_path)

        server = await asyncio.start_unix_server(
            self.handle_connection, path=socket_path
        )
        os.chmod(socket_path, 0o600)

        # Write PID file
        pid_file = get_worker_pid_file()
        with open(pid_file, "w") as f:
            f.write(str(os.getpid()))

        loop = asyncio.get_event_loop()
        stop = loop.create_future()

        def _shutdown(sig):
            if not stop.done():
                stop.set_result(sig)

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, _shutdown, sig)

        logger.info("Worker listening on %s (PID %d)", socket_path, os.getpid())

        try:
            await stop
        finally:
            server.close()
            await server.wait_closed()
            await self._client.aclose()
            if os.path.exists(socket_path):
                os.unlink(socket_path)
            if os.path.exists(pid_file):
                os.unlink(pid_file)
            logger.info("Worker shut down cleanly")


def run_worker():
    """Entry point for the worker daemon."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    worker = Worker()
    asyncio.run(worker.run())


if __name__ == "__main__":
    run_worker()
