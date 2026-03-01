"""Text normalization for TTS pronunciation.

Two-layer normalization:
1. Structural filters — regex patterns for file paths, CLI flags, etc.
2. Dictionary lookup — term -> phonetic replacements from pronunciations.yml

Moved from ~/Toolkit/lib/tts/normalize.py into the gateway package.
Dictionary path is configurable via TtsConfig.pronunciations.
"""

import re
from pathlib import Path

_compiled: list[tuple[re.Pattern, str]] | None = None
_dict_path: Path | None = None


def configure(pronunciations_path: str | None = None) -> None:
    """Set the pronunciations dictionary path. Call before first normalize()."""
    global _dict_path, _compiled
    if pronunciations_path:
        _dict_path = Path(pronunciations_path).expanduser()
    else:
        _dict_path = Path.home() / "Toolkit" / "voice" / "pronunciations.yml"
    _compiled = None  # Force reload


# ── Structural filters (applied before dictionary) ──────────────

def _simplify_file_path(match: re.Match) -> str:
    """Reduce file paths to just the filename, spoken naturally."""
    path = match.group(0)
    filename = path.rsplit("/", 1)[-1] if "/" in path else path
    if "." in filename:
        name = filename.rsplit(".", 1)[0]
        return f"{name} file"
    return filename


def _humanize_cli_flag(match: re.Match) -> str:
    """Make CLI flags speakable."""
    flag = match.group(0)
    clean = flag.lstrip("-")
    if "=" in clean:
        key, val = clean.split("=", 1)
        key = key.replace("-", " ")
        return f"{key} {val} option"
    clean = clean.replace("-", " ")
    return f"{clean} option"


_CODE_EXTS = (
    "js|ts|tsx|jsx|py|rb|go|rs|php|sh|yml|yaml|json|toml|md|"
    "css|html|sql|lua|swift|kt|java|c|cpp|h|vue|svelte"
)


def _apply_structural_filters(text: str) -> str:
    """Apply regex-based structural transformations."""
    # File paths
    text = re.sub(
        r"(?:(?<=\s)|^)(?:~/|\./)(?:[\w.@-]+/)*[\w.@-]+|(?:(?<=\s)|^)/(?:[\w.@-]+/)+[\w.@-]+",
        _simplify_file_path,
        text,
    )
    # CLI flags
    text = re.sub(
        r"(?:(?<=\s)|^)--[\w][\w-]*(?:=\S+)?|(?:(?<=\s)|^)-[a-zA-Z]\b",
        _humanize_cli_flag,
        text,
    )
    text = re.sub(r"\boption flag\b", "option", text)
    # Standalone filenames with extensions
    text = re.sub(
        rf"\b([\w][\w.-]*?)\.({_CODE_EXTS})\b",
        r"\1 file",
        text,
    )
    return text


# ── Dictionary loading ───────────────────────────────────────────

def _parse_simple(text: str) -> dict[str, str]:
    """Parse flat key: value pairs from YAML-like file without pyyaml."""
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if key and value:
            result[key] = value
    return result


def _load_dictionary() -> list[tuple[re.Pattern, str]]:
    """Load and compile pronunciation dictionary."""
    path = _dict_path or (Path.home() / "Toolkit" / "voice" / "pronunciations.yml")
    if not path.exists():
        return []

    raw = path.read_text()

    try:
        import yaml
        data = yaml.safe_load(raw)
    except ImportError:
        data = _parse_simple(raw)

    if not isinstance(data, dict):
        return []

    rules = []
    for term, replacement in data.items():
        pattern = re.compile(r"\b" + re.escape(str(term)) + r"\b", re.IGNORECASE)
        rules.append((pattern, str(replacement)))

    rules.sort(key=lambda r: len(r[0].pattern), reverse=True)
    return rules


# ── Public API ───────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Apply structural filters then pronunciation dictionary."""
    text = _apply_structural_filters(text)

    global _compiled
    if _compiled is None:
        _compiled = _load_dictionary()

    if _compiled:
        for pattern, replacement in _compiled:
            text = pattern.sub(replacement, text)

    return text


def reload():
    """Force reload of the pronunciation dictionary."""
    global _compiled
    _compiled = None
