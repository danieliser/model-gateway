# Model Gateway

A single-process gateway that routes AI model requests to local MLX models running in-process on Apple Silicon and cloud APIs (Anthropic, OpenAI) — all behind one OpenAI-compatible endpoint.

Local models run directly in the gateway process using Apple's Metal GPU. No subprocesses, no extra ports, no HTTP round-trips for local inference. Cloud models route through LiteLLM. External servers (LM Studio, Ollama) pass through via HTTP.

## Architecture

```
Client (curl / SDK / agent)
  │
  ▼
┌──────────────────────────────────────────┐
│              Model Gateway               │
│              :8800                        │
│                                          │
│  ┌───────────┐ ┌────────────┐ ┌────────┐│
│  │ LlmManager│ │EmbedManager│ │TtsMgr  ││  In-process (Metal GPU)
│  │ EngineCore│ │mlx-embed   │ │mlx-aud ││  No subprocess, no extra port
│  └───────────┘ └────────────┘ └────────┘│
│                                          │
│  ┌────────────┐ ┌────────────────────┐   │
│  │  LiteLLM   │ │  httpx             │   │  Cloud APIs / external servers
│  │ anthropic  │ │ lm-studio, ollama  │   │
│  │ openai     │ │ elevenlabs, google │   │
│  └────────────┘ └────────────────────┘   │
└──────────────────────────────────────────┘
```

- **MLX chat models** — loaded in-process via [vllm-mlx](https://github.com/vllm-project/vllm-mlx) `EngineCore` with continuous batching for concurrent requests
- **MLX embedding models** — loaded in-process via [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings)
- **MLX TTS models** — loaded in-process via [mlx-audio](https://github.com/Blaizzy/mlx-audio) (Kokoro, Chatterbox, etc.)
- **Cloud backends** — Anthropic and OpenAI (LiteLLM), ElevenLabs and Google Cloud TTS (httpx)
- **External backends** — LM Studio, Ollama proxied via httpx passthrough
- **Dynamic loading** — models load on first request, unload after idle timeout (default 15 min)

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended)

## Quick Start

```bash
# Clone and install
git clone https://github.com/danieliser/model-gateway.git
cd model-gateway
uv sync

# Create config
mkdir -p ~/.config/model-gateway
cp config.yml.example ~/.config/model-gateway/config.yml
# Edit to set your model paths and API keys

# Start the gateway
model-gateway start

# Send a request
curl http://localhost:8800/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-4b", "messages": [{"role": "user", "content": "Hello"}]}'
```

## Configuration

Config is loaded from (in order):
1. Explicit `--config` path
2. `$MODEL_GATEWAY_CONFIG` env var
3. `~/.config/model-gateway/config.yml`

```yaml
port: 8800
default_model: qwen3-4b
embedding_model: nomic-embed

models:
  qwen3-4b:
    backend: mlx
    model_id: mlx-community/Qwen3-4B-Instruct-4bit
  nomic-embed:
    backend: mlx-embed
    model_id: ~/models/nomicai-modernbert-embed-base-bf16
  haiku:
    backend: anthropic
    model_id: claude-haiku-4-5-20251001
    api_key_env: ANTHROPIC_API_KEY
  gpt-4o:
    backend: openai
    model_id: gpt-4o
    api_key_env: OPENAI_API_KEY
  kokoro:
    backend: mlx-audio
    model_id: kokoro-82m-bf16
    idle_timeout: 600
  chatterbox:
    backend: mlx-audio
    model_id: chatterbox-turbo-fp16
    idle_timeout: 300

backends:
  mlx:
    enabled: true
  mlx-embed:
    enabled: true
  anthropic:
    enabled: true
    api_key_env: ANTHROPIC_API_KEY
  openai:
    enabled: true
    api_key_env: OPENAI_API_KEY
  mlx-audio:
    enabled: true
  lm-studio:
    enabled: true
    host: http://localhost:1234
  ollama:
    enabled: false
    host: http://localhost:11434
  elevenlabs:
    enabled: false
    api_key_env: ELEVENLABS_API_KEY
  google-tts:
    enabled: false
    api_key_env: GOOGLE_CLOUD_API_KEY

# TTS settings
tts:
  pronunciations: ~/voice/pronunciations.yml  # optional
  cache_dir: ~/cache/tts
  default_voice: af_heart
  default_lang_code: a
  default_speed: 1.0

# Route "auto" model requests by task type (via X-Task-Type header)
task_routing:
  extraction: qwen3-4b
  classification: qwen3-4b
  coding: sonnet
  reasoning: sonnet
  general: haiku

# Idle timeout in seconds (default 900 = 15 min)
idle_timeout: 900
```

### Model Options

| Field | Description |
|-------|-------------|
| `backend` | Backend to use: `mlx`, `mlx-embed`, `mlx-audio`, `llama-cpp`, `lm-studio`, `ollama`, `anthropic`, `openai`, `elevenlabs`, `google-tts` |
| `model_id` | HuggingFace repo ID or local path (for local backends) / API model ID (for cloud) |
| `model_path` | Alternative to `model_id` for local path |
| `api_key_env` | Environment variable name containing the API key |
| `pin` | If `true`, model is never auto-unloaded on idle |
| `idle_timeout` | Per-model idle timeout override (seconds) |

## CLI Reference

```
model-gateway start [--port N] [--foreground]    Start the gateway server
model-gateway stop                                Stop the gateway server
model-gateway restart                             Restart the gateway
model-gateway status                              Show gateway status
model-gateway health                              Check health of all backends

model-gateway chat "prompt" [-m model] [-s system] [--stream/--no-stream] [--json]
model-gateway embed "text" [-m model] [--file path]
model-gateway complete "prompt" [-m model] [--max-tokens N]
model-gateway speak "text" [-m model] [--voice V] [--speed N] [-o file]
model-gateway test <alias>                        Send a test prompt through the gateway

model-gateway models                              List models and their status
model-gateway switch <alias>                      Switch the default model

model-gateway config show                         Print resolved config (keys redacted)
model-gateway config edit                         Open config in $EDITOR
model-gateway config validate                     Validate config file

model-gateway logs [-f] [-n lines]                Show gateway logs
```

## API Endpoints

### Chat Completions
```
POST /v1/chat/completions
```
OpenAI-compatible. Supports `stream: true` for SSE streaming.

### Embeddings
```
POST /v1/embeddings
```
OpenAI-compatible embedding generation.

### Text-to-Speech
```
POST /v1/audio/speech
```
OpenAI-compatible TTS. Returns WAV or MP3 audio bytes.

```bash
curl http://localhost:8800/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "kokoro", "input": "Hello world", "voice": "af_heart"}' \
  -o output.wav
```

Accepts `model` as a config alias (`kokoro`), model ID (`kokoro-82m-bf16`), or path. Optional params: `speed`, `lang_code`, `exaggeration`, `conds_path`, `ref_audio`, `response_format` (`wav`/`mp3`).

```
GET  /v1/audio/cache/stats     Cache statistics
POST /v1/audio/cache/prune     Prune expired cache entries
```

### Health
```
GET /health
```
Returns gateway status, loaded models (with `port: null` for in-process), and available models.

### Model Management
```
GET  /v1/models                      List all models
POST /gateway/switch                 Switch default model
POST /gateway/models/load            Explicitly load a model
POST /gateway/models/unload          Unload a model from memory
GET  /gateway/config                 View config (keys redacted)
```

## Supported Backends

| Backend | Type | Description |
|---------|------|-------------|
| `mlx` | In-process | MLX chat models via vllm-mlx EngineCore (continuous batching) |
| `mlx-embed` | In-process | MLX embedding models via mlx-embeddings |
| `llama-cpp` | Subprocess | llama.cpp server (llama-server binary) |
| `lm-studio` | External | LM Studio API passthrough |
| `ollama` | External | Ollama API passthrough |
| `mlx-audio` | In-process | MLX TTS models via mlx-audio (Kokoro, Chatterbox, etc.) |
| `anthropic` | Cloud | Anthropic API via LiteLLM |
| `openai` | Cloud | OpenAI API via LiteLLM |
| `elevenlabs` | Cloud | ElevenLabs TTS API |
| `google-tts` | Cloud | Google Cloud Text-to-Speech API |

## Development

```bash
# Install with dev dependencies
uv sync --group dev

# Run tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_llm_server.py -v
```

## License

MIT
