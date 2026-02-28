# Research: Eliminating LiteLLM Router Overhead for Local Backends
**Date:** 2026-02-22
**Question:** How to reduce the 125ms LiteLLM routing overhead to achieve sub-100ms gateway latency while keeping multi-backend routing?
**Recommendation:** Hybrid approach — bypass LiteLLM for local backends with a thin httpx pass-through proxy, keep LiteLLM only for cloud backends.

## Context & Constraints

- vllm-mlx direct: **48ms** per request
- Through gateway (LiteLLM Router): **174ms** — 125ms overhead
- Must preserve multi-backend routing (mlx, anthropic, openai, ollama)
- Must support streaming
- Single-developer project, simplicity > enterprise features

## Where the 125ms Goes

LiteLLM Router (`router.acompletion()`) does far more than route:
1. **Request serialization** — Python dict → LiteLLM internal format (~5ms)
2. **HTTP call via httpx** — New connection or pool lookup to localhost:8801 (~10-15ms)
3. **Response deserialization** — Raw JSON → LiteLLM ModelResponse object (~10ms)
4. **LiteLLM middleware** — Logging, retries, fallback logic, model mapping (~5-10ms)
5. **Import/initialization tax** — LiteLLM imports 100+ modules on first call (~50-80ms amortized)

LiteLLM's own benchmarks show 2-14ms overhead for their proxy server at scale, but that's the optimized Go/Rust path. The Python Router SDK doing full request/response round-trips adds substantially more.

## Options Evaluated

### Option 1: Thin httpx Reverse Proxy (for local backends)
- **Confidence:** High
- **What it is:** For local backends (mlx, ollama, llama.cpp), bypass LiteLLM entirely. Forward raw HTTP bytes via `httpx.AsyncClient` with streaming. Keep LiteLLM only for cloud backends (anthropic, openai) where it adds real value (auth, retries, model mapping).
- **Strengths:**
  - Estimated overhead: 2-5ms (httpx connection pool + async forward)
  - Preserves streaming natively (proxy raw SSE chunks)
  - No serialization/deserialization — pass bytes through
  - Simple implementation (~50 lines in proxy.py)
  - No new dependencies
- **Weaknesses:**
  - Two code paths (local vs cloud) increases complexity slightly
  - Lose LiteLLM's unified response format for local backends (but vllm-mlx already returns OpenAI format)
- **Expected latency:** 48ms + 2-5ms = **50-53ms** through gateway

### Option 2: fastapi-proxy-lib
- **Confidence:** Medium
- **What it is:** Drop-in reverse proxy library for FastAPI/Starlette. Uses httpx with forced keep-alive connections.
- **Strengths:**
  - Battle-tested library, handles edge cases (WebSocket, disconnects)
  - One-liner setup: `reverse_http_app(base_url="http://localhost:8801")`
  - Streaming support built-in
- **Weaknesses:**
  - Another dependency for a simple problem
  - No published latency benchmarks
  - Overkill for proxying to localhost
- **Expected latency:** ~50-55ms (similar to raw httpx)

### Option 3: Embed vllm-mlx In-Process
- **Confidence:** Low
- **What it is:** Load vllm-mlx as a Python library inside the FastAPI process, eliminating HTTP entirely.
- **Strengths:**
  - Zero network overhead — direct function call
  - Theoretical minimum latency (~48ms or less)
- **Weaknesses:**
  - Loses process isolation (crash takes down gateway)
  - Loses model switching without restart
  - vllm-mlx may not expose a clean library API
  - Blocks the event loop during inference
  - High implementation effort (2-3 days)
- **Expected latency:** ~45-48ms

### Option 4: Replace LiteLLM with Bifrost
- **Confidence:** Low
- **What it is:** Maxim AI's Bifrost claims 11µs overhead at 5K RPS. Rust-based.
- **Strengths:**
  - Near-zero overhead
  - OpenAI-compatible API
- **Weaknesses:**
  - Separate service to run and manage
  - Cloud-hosted or complex self-host setup
  - Overkill for single-machine, single-user scenario
  - Adds operational complexity
- **Expected latency:** ~48-50ms

## Comparison Matrix

| Criteria | httpx pass-through | fastapi-proxy-lib | In-process embed | Bifrost |
|----------|-------------------|-------------------|-----------------|---------|
| Expected latency | 50-53ms | 50-55ms | 45-48ms | 48-50ms |
| Implementation effort | 2-3 hours | 1-2 hours | 2-3 days | 4-6 hours |
| New dependencies | None | 1 (fastapi-proxy-lib) | Complex | 1 (external service) |
| Streaming support | Yes (raw proxy) | Yes (built-in) | Manual | Yes |
| Multi-backend routing | Yes (hybrid) | Manual routing | No | Yes |
| Risk | Low | Low | High | Medium |
| Cloud backend support | Via LiteLLM | Manual | N/A | Built-in |

## Recommendation

**Option 1: Thin httpx pass-through for local backends, keep LiteLLM for cloud.**

The implementation is straightforward: in `proxy.py`, check if the target backend is local. If yes, forward the raw request via `httpx.AsyncClient.stream()` to `http://localhost:{port}/v1/chat/completions` and return a `StreamingResponse` with the raw bytes. If the backend is cloud (anthropic, openai), use the existing LiteLLM Router path.

This gives you:
- **50-53ms** through the gateway for local models (down from 174ms)
- Full LiteLLM features preserved for cloud providers (auth, model mapping, retries)
- No new dependencies
- ~50 lines of code change

**What would change this recommendation:** If you add many cloud providers with complex routing needs, a unified proxy like Bifrost becomes more attractive. If you need sub-50ms, in-process embedding is the only path but the tradeoffs are severe.

### Implementation Sketch

```python
# In proxy.py — add to ProxyManager

async def _local_passthrough(self, backend_url: str, payload: dict, stream: bool) -> Any:
    """Bypass LiteLLM — raw httpx proxy to local backend."""
    url = f"{backend_url}/chat/completions"
    if stream:
        req = self._http_client.build_request("POST", url, json=payload)
        resp = await self._http_client.send(req, stream=True)
        return StreamingResponse(
            resp.aiter_raw(),
            media_type="text/event-stream",
            headers={"X-Gateway-Proxy": "passthrough"},
        )
    else:
        resp = await self._http_client.post(url, json=payload)
        return JSONResponse(content=resp.json())

async def completion(self, model, messages, stream=False, **kwargs):
    model_cfg = self._config.models.get(model)
    if model_cfg and model_cfg.backend not in _CLOUD_BACKENDS:
        api_base = self._backend_urls.get(model_cfg.backend)
        if api_base:
            payload = {"model": model_cfg.model_id, "messages": messages, "stream": stream, **kwargs}
            return await self._local_passthrough(api_base, payload, stream)
    # Fall through to LiteLLM for cloud backends
    return await self._router.acompletion(model=model, messages=messages, stream=stream, **kwargs)
```

Key detail: create `self._http_client = httpx.AsyncClient(timeout=120.0)` once in `__init__` — connection pooling eliminates per-request connection setup.

## Sources

- [LiteLLM Benchmarks](https://docs.litellm.ai/docs/benchmarks) — P50: 2-12ms overhead depending on instance count
- [LiteLLM v1.77.7 Release](https://docs.litellm.ai/release_notes/v1-77-7) — 2.9x lower median latency improvements
- [fastapi-proxy-lib](https://github.com/WSH032/fastapi-proxy-lib) — Forced keep-alive, streaming support, minimal setup
- [Top LLM Gateways 2025](https://agenta.ai/blog/top-llm-gateways) — Bifrost 11µs overhead claim
- [Top 5 LiteLLM Alternatives](https://www.getmaxim.ai/articles/top-5-litellm-alternatives-in-2025/) — Bifrost, Helicone comparisons
- [FastAPI Proxy Discussion](https://github.com/fastapi/fastapi/discussions/9599) — httpx streaming proxy patterns
- [LiteLLM Review 2026](https://www.truefoundry.com/blog/a-detailed-litellm-review-features-pricing-pros-and-cons-2026) — Performance issues at scale
