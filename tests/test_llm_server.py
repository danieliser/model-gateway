"""Tests for model_gateway.llm_server module."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import model_gateway.llm_server as llm_mod
from model_gateway.llm_server import LlmManager


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _mock_engine():
    """Create a mock EngineCore."""
    engine = MagicMock()
    engine.start = AsyncMock()
    engine.stop = AsyncMock()
    engine.close = MagicMock()
    engine.add_request = AsyncMock(return_value="req-001")
    return engine


def _mock_output(text="Hello", new_text="Hello", finished=True, finish_reason="stop",
                 prompt_tokens=5, completion_tokens=3):
    output = MagicMock()
    output.output_text = text
    output.new_text = new_text
    output.finished = finished
    output.finish_reason = finish_reason
    output.prompt_tokens = prompt_tokens
    output.completion_tokens = completion_tokens
    return output


def _mock_tokenizer():
    tok = MagicMock()
    tok.apply_chat_template.return_value = "<|user|>Hi<|assistant|>"
    return tok


@pytest.fixture(autouse=True)
def _patch_imports(monkeypatch):
    """Ensure lazy imports are resolved with mocks so tests don't need real MLX."""
    monkeypatch.setattr(llm_mod, "mlx_load", MagicMock())
    monkeypatch.setattr(llm_mod, "EngineCore", MagicMock())
    monkeypatch.setattr(llm_mod, "EngineConfig", MagicMock())
    monkeypatch.setattr(llm_mod, "SamplingParams", MagicMock())


# ---------------------------------------------------------------------------
# Test: load / unload / is_loaded / touch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_load_and_is_loaded(monkeypatch):
    manager = LlmManager()

    mock_model = MagicMock()
    mock_tok = _mock_tokenizer()
    mock_eng = _mock_engine()

    monkeypatch.setattr(llm_mod, "mlx_load", MagicMock(return_value=(mock_model, mock_tok)))
    monkeypatch.setattr(llm_mod, "EngineCore", MagicMock(return_value=mock_eng))

    await manager.load("test-model", "path/to/model")

    assert manager.is_loaded("test-model")
    assert not manager.is_loaded("other-model")
    mock_eng.start.assert_awaited_once()


@pytest.mark.asyncio
async def test_load_already_loaded_touches(monkeypatch):
    manager = LlmManager()

    mock_model = MagicMock()
    mock_tok = _mock_tokenizer()
    mock_eng = _mock_engine()

    monkeypatch.setattr(llm_mod, "mlx_load", MagicMock(return_value=(mock_model, mock_tok)))
    monkeypatch.setattr(llm_mod, "EngineCore", MagicMock(return_value=mock_eng))

    await manager.load("test-model", "path/to/model")
    old_time = manager.get_last_used("test-model")

    await manager.load("test-model", "path/to/model")
    new_time = manager.get_last_used("test-model")

    assert new_time >= old_time
    assert mock_eng.start.await_count == 1


@pytest.mark.asyncio
async def test_unload(monkeypatch):
    manager = LlmManager()

    mock_model = MagicMock()
    mock_tok = _mock_tokenizer()
    mock_eng = _mock_engine()

    monkeypatch.setattr(llm_mod, "mlx_load", MagicMock(return_value=(mock_model, mock_tok)))
    monkeypatch.setattr(llm_mod, "EngineCore", MagicMock(return_value=mock_eng))

    await manager.load("test-model", "path/to/model")
    assert manager.is_loaded("test-model")

    result = await manager.unload("test-model")

    assert result is True
    assert not manager.is_loaded("test-model")
    mock_eng.stop.assert_awaited_once()
    mock_eng.close.assert_called_once()


@pytest.mark.asyncio
async def test_unload_not_loaded():
    manager = LlmManager()
    result = await manager.unload("nonexistent")
    assert result is False


def test_touch_updates_timestamp():
    manager = LlmManager()
    manager._models["test"] = MagicMock()
    manager._models["test"].last_used = time.monotonic() - 100

    old = manager._models["test"].last_used
    manager.touch("test")
    assert manager._models["test"].last_used > old


def test_get_last_used_none_if_not_loaded():
    manager = LlmManager()
    assert manager.get_last_used("nonexistent") is None


# ---------------------------------------------------------------------------
# Test: generate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_returns_openai_format(monkeypatch):
    manager = LlmManager()

    mock_model = MagicMock()
    mock_tok = _mock_tokenizer()
    mock_eng = _mock_engine()

    final_output = _mock_output(text="Hello world", finished=True)

    async def fake_stream(request_id):
        yield final_output

    mock_eng.stream_outputs = fake_stream
    mock_eng.add_request = AsyncMock(return_value="req-001")

    monkeypatch.setattr(llm_mod, "mlx_load", MagicMock(return_value=(mock_model, mock_tok)))
    monkeypatch.setattr(llm_mod, "EngineCore", MagicMock(return_value=mock_eng))

    await manager.load("test-model", "path/to/model")

    messages = [{"role": "user", "content": "Hi"}]
    result = await manager.generate("test-model", messages, max_tokens=100)

    assert result["object"] == "chat.completion"
    assert result["choices"][0]["message"]["role"] == "assistant"
    assert result["choices"][0]["message"]["content"] == "Hello world"
    assert result["choices"][0]["finish_reason"] == "stop"
    assert "usage" in result
    assert result["usage"]["prompt_tokens"] == 5
    assert result["usage"]["completion_tokens"] == 3
    assert result["id"].startswith("chatcmpl-")

    mock_tok.apply_chat_template.assert_called_once_with(
        messages, tokenize=False, add_generation_prompt=True,
    )


@pytest.mark.asyncio
async def test_generate_not_loaded_raises():
    manager = LlmManager()
    with pytest.raises(RuntimeError, match="not loaded"):
        await manager.generate("nonexistent", [{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------------------
# Test: stream_generate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_generate_yields_chunks(monkeypatch):
    manager = LlmManager()

    mock_model = MagicMock()
    mock_tok = _mock_tokenizer()
    mock_eng = _mock_engine()

    chunk1 = _mock_output(new_text="Hello", finished=False)
    chunk2 = _mock_output(new_text=" world", finished=False)
    final = _mock_output(new_text="", finished=True, finish_reason="stop")

    async def fake_stream(request_id):
        yield chunk1
        yield chunk2
        yield final

    mock_eng.stream_outputs = fake_stream

    monkeypatch.setattr(llm_mod, "mlx_load", MagicMock(return_value=(mock_model, mock_tok)))
    monkeypatch.setattr(llm_mod, "EngineCore", MagicMock(return_value=mock_eng))

    await manager.load("test-model", "path/to/model")

    messages = [{"role": "user", "content": "Hi"}]
    chunks = []
    async for chunk in manager.stream_generate("test-model", messages):
        chunks.append(chunk)

    assert len(chunks) == 3
    assert chunks[0]["object"] == "chat.completion.chunk"
    assert chunks[0]["choices"][0]["delta"]["content"] == "Hello"
    assert chunks[0]["choices"][0]["finish_reason"] is None
    assert chunks[1]["choices"][0]["delta"]["content"] == " world"
    assert chunks[2]["choices"][0]["finish_reason"] == "stop"
    assert chunks[2]["choices"][0]["delta"] == {}

    # All share same completion ID
    assert chunks[0]["id"] == chunks[1]["id"] == chunks[2]["id"]


@pytest.mark.asyncio
async def test_stream_generate_not_loaded_raises():
    manager = LlmManager()
    with pytest.raises(RuntimeError, match="not loaded"):
        async for _ in manager.stream_generate("nonexistent", []):
            pass


# ---------------------------------------------------------------------------
# Test: sampling params
# ---------------------------------------------------------------------------

def test_build_sampling_params(monkeypatch):
    manager = LlmManager()
    mock_sp = MagicMock()
    monkeypatch.setattr(llm_mod, "SamplingParams", mock_sp)

    manager._build_sampling_params(
        max_tokens=100, temperature=0.7, top_p=0.9, stop=["<|end|>"]
    )
    mock_sp.assert_called_once_with(
        max_tokens=100, temperature=0.7, top_p=0.9, stop=["<|end|>"]
    )


def test_build_sampling_params_string_stop(monkeypatch):
    manager = LlmManager()
    mock_sp = MagicMock()
    monkeypatch.setattr(llm_mod, "SamplingParams", mock_sp)

    manager._build_sampling_params(stop="<|end|>")
    mock_sp.assert_called_once_with(stop=["<|end|>"])
