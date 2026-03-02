"""Audio caching layer for TTS.

Cache key is SHA256 of the canonical parameters (text, model_id, voice,
lang_code, speed). Audio stored as {hash}.wav with metadata in index.json.

Moved from ~/Toolkit/lib/tts/cache.py into the gateway package.
Cache dir is configurable via configure().
"""

import hashlib
import json
import time
from pathlib import Path

# Defaults — overridable via configure()
_cache_dir: Path = Path.home() / "Toolkit" / "cache" / "tts"
_cache_index: Path = _cache_dir / "index.json"
_cache_max_mb: int = 500
_cache_max_age_days: int = 30


def configure(
    cache_dir: str | None = None,
    max_mb: int = 500,
    max_age_days: int = 30,
) -> None:
    """Set cache configuration. Call before first use."""
    global _cache_dir, _cache_index, _cache_max_mb, _cache_max_age_days
    if cache_dir:
        _cache_dir = Path(cache_dir).expanduser()
    _cache_index = _cache_dir / "index.json"
    _cache_max_mb = max_mb
    _cache_max_age_days = max_age_days


def _cache_key(
    text: str, model_id: str, voice: str, lang_code: str, speed: float,
    conds_path: str | None = None,
) -> str:
    """Deterministic hash for a set of TTS parameters."""
    params: dict = {"text": text, "model_id": model_id, "voice": voice, "lang_code": lang_code, "speed": speed}
    if conds_path:
        params["conds_path"] = conds_path
    payload = json.dumps(params, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_index() -> dict:
    if _cache_index.exists():
        try:
            return json.loads(_cache_index.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_index(index: dict) -> None:
    _cache_dir.mkdir(parents=True, exist_ok=True)
    _cache_index.write_text(json.dumps(index, indent=2))


def cache_lookup(
    text: str, model_id: str, voice: str, lang_code: str, speed: float,
    conds_path: str | None = None,
) -> Path | None:
    """Return cached audio path if it exists, otherwise None."""
    key = _cache_key(text, model_id, voice, lang_code, speed, conds_path)
    index = _load_index()
    entry = index.get(key)
    if not entry:
        return None

    path = _cache_dir / f"{key}.wav"
    if not path.exists():
        del index[key]
        _save_index(index)
        return None

    entry["last_accessed"] = time.time()
    entry["access_count"] = entry.get("access_count", 0) + 1
    _save_index(index)
    return path


def cache_store(
    text: str,
    model_id: str,
    voice: str,
    lang_code: str,
    speed: float,
    audio_bytes: bytes,
    duration_ms: float = 0,
    conds_path: str | None = None,
) -> Path:
    """Store audio bytes in cache. Returns the cache file path."""
    _cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(text, model_id, voice, lang_code, speed, conds_path)
    path = _cache_dir / f"{key}.wav"
    path.write_bytes(audio_bytes)

    index = _load_index()
    index[key] = {
        "text": text,
        "model_id": model_id,
        "voice": voice,
        "lang_code": lang_code,
        "speed": speed,
        "created_at": time.time(),
        "last_accessed": time.time(),
        "access_count": 1,
        "file_size_bytes": len(audio_bytes),
        "audio_duration_ms": duration_ms,
    }
    _save_index(index)
    return path


def cache_prune() -> dict:
    """Run eviction rules. Returns stats about what was pruned."""
    index = _load_index()
    now = time.time()
    max_age_secs = _cache_max_age_days * 86400
    max_bytes = _cache_max_mb * 1024 * 1024

    pruned_age = 0
    pruned_lru = 0

    # Pass 1: age + low access eviction
    keys_to_remove = []
    for key, entry in index.items():
        age = now - entry.get("created_at", now)
        if age > max_age_secs and entry.get("access_count", 0) < 3:
            keys_to_remove.append(key)

    for key in keys_to_remove:
        path = _cache_dir / f"{key}.wav"
        path.unlink(missing_ok=True)
        del index[key]
        pruned_age += 1

    # Pass 2: LRU eviction if over size budget
    total_size = sum(e.get("file_size_bytes", 0) for e in index.values())
    if total_size > max_bytes:
        target = int(max_bytes * 0.8)
        sorted_entries = sorted(index.items(), key=lambda x: x[1].get("last_accessed", 0))
        for key, entry in sorted_entries:
            if total_size <= target:
                break
            path = _cache_dir / f"{key}.wav"
            total_size -= entry.get("file_size_bytes", 0)
            path.unlink(missing_ok=True)
            del index[key]
            pruned_lru += 1

    _save_index(index)
    return {"pruned_age": pruned_age, "pruned_lru": pruned_lru, "remaining": len(index)}


def cache_stats() -> dict:
    """Return cache statistics."""
    index = _load_index()
    total_size = sum(e.get("file_size_bytes", 0) for e in index.values())
    total_accesses = sum(e.get("access_count", 0) for e in index.values())

    top_phrases = sorted(index.values(), key=lambda x: x.get("access_count", 0), reverse=True)[:10]

    return {
        "entries": len(index),
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "total_accesses": total_accesses,
        "max_size_mb": _cache_max_mb,
        "top_phrases": [
            {"text": p["text"][:60], "voice": p["voice"], "accesses": p["access_count"]}
            for p in top_phrases
        ],
    }


def cache_list(sort_by: str = "access") -> list[dict]:
    """List all cache entries, sorted by given field."""
    index = _load_index()
    sort_keys = {
        "access": lambda x: x[1].get("last_accessed", 0),
        "age": lambda x: x[1].get("created_at", 0),
        "size": lambda x: x[1].get("file_size_bytes", 0),
    }
    sort_fn = sort_keys.get(sort_by, sort_keys["access"])
    entries = sorted(index.items(), key=sort_fn, reverse=True)
    return [
        {
            "hash": k[:12],
            "text": v["text"][:50],
            "voice": v["voice"],
            "size_kb": round(v.get("file_size_bytes", 0) / 1024, 1),
            "accesses": v.get("access_count", 0),
            "age_hours": round((time.time() - v.get("created_at", time.time())) / 3600, 1),
        }
        for k, v in entries
    ]


def cache_clear(older_than_days: int | None = None, model_id: str | None = None) -> int:
    """Selectively clear cache entries. Returns count of removed entries."""
    index = _load_index()
    now = time.time()
    removed = 0

    keys_to_remove = []
    for key, entry in index.items():
        should_remove = False
        if older_than_days is not None:
            age_days = (now - entry.get("created_at", now)) / 86400
            if age_days > older_than_days:
                should_remove = True
        if model_id is not None and entry.get("model_id") == model_id:
            should_remove = True
        if older_than_days is None and model_id is None:
            should_remove = True

        if should_remove:
            keys_to_remove.append(key)

    for key in keys_to_remove:
        path = _cache_dir / f"{key}.wav"
        path.unlink(missing_ok=True)
        del index[key]
        removed += 1

    _save_index(index)
    return removed
