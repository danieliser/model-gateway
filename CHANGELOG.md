# Changelog

All notable changes to model-gateway will be documented in this file.

## [0.3.0] - 2026-03-01

### Added
- **TTS backend (`mlx-audio`)** — In-process text-to-speech via MLX-Audio on Apple Silicon
- **`POST /v1/audio/speech`** — OpenAI-compatible TTS endpoint returning WAV or MP3
- **TtsManager** — Model lifecycle (lazy load, idle timeout, unload) matching EmbedManager pattern
- **Text normalization** — Pronunciation dictionary for natural speech output
- **Audio cache** — SHA256-keyed WAV cache with configurable size/age eviction
- **Cloud TTS routing** — ElevenLabs and Google Cloud TTS via proxy
- **Model ID resolution** — `/v1/audio/speech` accepts config alias, model_id, or filesystem path
- **`speak` CLI command** — Generate and play TTS audio from the terminal
- **Worker `tts` request type** — Socket-based TTS for integrations
- **`tts` config section** — Pronunciations path, cache dir, default voice/speed/lang
- **`config.yml` example** — Shipped as reference config in repo

## [0.2.0] - 2026-02-28

### Added
- **In-process MLX chat** via vllm-mlx EngineCore with continuous batching
- README and config example

### Fixed
- Strip EOS/stop tokens from MLX chat output

## [0.1.0] - 2026-02-27

### Added
- Initial gateway: FastAPI server on port 8800
- Config schema and loader (`config.py`)
- Backend lifecycle manager with dynamic model loading/unloading
- In-process MLX embeddings via mlx-embeddings
- LiteLLM proxy for Anthropic and OpenAI cloud APIs
- httpx passthrough for LM Studio and Ollama
- CLI with chat, embed, complete, models, status, config commands
- Persistent Unix socket worker for fast CLI routing
- Task-based routing via `X-Task-Type` header
- Idle timeout sweep for automatic model unloading
