"""In-process MLX-Audio TTS inference.

Manages TTS model loading/unloading and provides a generate() function
that returns raw WAV audio bytes. No subprocess, no extra port — runs
inside the gateway process.

Mirrors the EmbedManager pattern from embed_server.py.
"""

from __future__ import annotations

import io
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class _LoadedTtsModel:
    model: Any  # mlx_audio model object
    model_name: str  # basename of model path
    model_path: str  # full resolved path
    last_used: float = field(default_factory=time.monotonic)


class TtsManager:
    """Manages in-process MLX-Audio TTS models."""

    def __init__(self) -> None:
        self._models: dict[str, _LoadedTtsModel] = {}

    @staticmethod
    def _resolve_model_path(model_id: str) -> str:
        """Resolve a model ID to a filesystem path.

        Resolution chain:
        1. If absolute path, use directly
        2. Expand ~ paths
        3. Check ~/Toolkit/models/index.yml for path mapping
        4. Fall back to ~/Toolkit/models/{model_id}/
        """
        if model_id.startswith("/"):
            return model_id

        expanded = os.path.expanduser(model_id)
        if os.path.isdir(expanded):
            return expanded

        models_dir = os.path.expanduser("~/Toolkit/models")
        index_path = os.path.join(models_dir, "index.yml")
        if os.path.exists(index_path):
            try:
                import yaml
                with open(index_path) as f:
                    index = yaml.safe_load(f) or {}
                entry = (index.get("models") or {}).get(model_id, {})
                if isinstance(entry, dict) and entry.get("path"):
                    candidate = os.path.join(models_dir, entry["path"])
                    if os.path.isdir(candidate):
                        return candidate
            except Exception:
                pass

        # Fall back to models_dir / model_id
        fallback = os.path.join(models_dir, model_id)
        if os.path.isdir(fallback):
            return fallback

        return expanded  # Return whatever we have, let load_model() handle errors

    def is_loaded(self, model_alias: str) -> bool:
        return model_alias in self._models

    def load(self, model_alias: str, model_path: str) -> None:
        """Load a TTS model into memory.

        model_path can be:
        - An absolute path
        - A ~ path (expanded)
        - A short model ID (resolved via ~/Toolkit/models/index.yml)
        """
        if model_alias in self._models:
            self._models[model_alias].last_used = time.monotonic()
            return

        resolved = self._resolve_model_path(model_path)
        logger.info("Loading TTS model '%s' from %s", model_alias, resolved)

        from mlx_audio.tts.utils import load_model
        model = load_model(Path(resolved))

        self._models[model_alias] = _LoadedTtsModel(
            model=model,
            model_name=os.path.basename(resolved.rstrip("/")),
            model_path=resolved,
        )
        logger.info("TTS model '%s' loaded", model_alias)

    def unload(self, model_alias: str) -> bool:
        """Unload a TTS model from memory. Returns True if unloaded."""
        info = self._models.pop(model_alias, None)
        if info is None:
            return False
        del info.model
        logger.info("TTS model '%s' unloaded", model_alias)
        return True

    def touch(self, model_alias: str) -> None:
        """Update last_used timestamp."""
        info = self._models.get(model_alias)
        if info:
            info.last_used = time.monotonic()

    def get_last_used(self, model_alias: str) -> float | None:
        info = self._models.get(model_alias)
        return info.last_used if info else None

    def generate(
        self,
        model_alias: str,
        text: str,
        *,
        voice: str = "af_heart",
        speed: float = 1.0,
        lang_code: str = "a",
        exaggeration: Optional[float] = None,
        instruct: Optional[str] = None,
        conds_path: Optional[str] = None,
        ref_audio: Optional[str] = None,
        response_format: str = "wav",
    ) -> bytes:
        """Generate TTS audio. Returns raw audio bytes (WAV or MP3).

        Parameters
        ----------
        model_alias : str
            The alias of a loaded TTS model.
        text : str
            Text to synthesize.
        voice : str
            Voice ID (e.g. "af_heart"). Passed to the model's generate().
        speed : float
            Playback speed multiplier (1.0 = normal).
        lang_code : str
            Language code (e.g. "a" for American English).
        exaggeration : float, optional
            Expressiveness level (Chatterbox: 0.0-1.0).
        instruct : str, optional
            Voice description text (Qwen3 VoiceDesign).
        conds_path : str, optional
            Path to pre-extracted voice conditionals (.conds/.safetensors).
        ref_audio : str, optional
            Path to reference audio for voice cloning.
        response_format : str
            Output format: "wav" (default) or "mp3".
        """
        info = self._models.get(model_alias)
        if info is None:
            raise RuntimeError(f"TTS model '{model_alias}' not loaded")

        info.last_used = time.monotonic()

        # Build kwargs for model.generate() — different models accept different params
        gen_kwargs: dict[str, Any] = {}
        if voice:
            gen_kwargs["voice"] = voice
        if speed != 1.0:
            gen_kwargs["speed"] = speed
        if lang_code:
            gen_kwargs["lang_code"] = lang_code
        if exaggeration is not None:
            gen_kwargs["exaggeration"] = exaggeration
        if instruct:
            gen_kwargs["instruct"] = instruct
        if conds_path:
            gen_kwargs["conds_path"] = conds_path
        if ref_audio:
            gen_kwargs["ref_audio"] = ref_audio

        # Generate audio segments — mlx_audio yields result objects
        # with .audio (numpy array) and .sample_rate attributes
        import numpy as np
        audio_chunks = []
        sample_rate = 24000

        for result in info.model.generate(text, **gen_kwargs):
            if hasattr(result, "audio"):
                audio_chunks.append(np.array(result.audio))
                if hasattr(result, "sample_rate") and result.sample_rate:
                    sample_rate = result.sample_rate
            else:
                # Raw array fallback
                audio_chunks.append(np.array(result))

        if not audio_chunks:
            raise RuntimeError("No audio generated")

        audio = np.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]

        # Ensure 1D for mono audio
        if audio.ndim > 1:
            audio = audio.flatten()

        # Encode to WAV
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
        wav_bytes = buf.getvalue()

        if response_format == "mp3":
            wav_bytes = self._wav_to_mp3(wav_bytes)

        return wav_bytes

    @staticmethod
    def _wav_to_mp3(wav_bytes: bytes) -> bytes:
        """Convert WAV bytes to MP3 via ffmpeg. Falls back to WAV on error."""
        import subprocess
        try:
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", "pipe:0", "-f", "mp3", "-q:a", "2", "pipe:1"],
                input=wav_bytes,
                capture_output=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout:
                return result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        logger.warning("MP3 conversion failed, returning WAV")
        return wav_bytes
