from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

DEFAULT_PROJECTION_CONFIG: dict[str, Any] = {
    "playlist_id": "huaihai-75-v1",
    "interval_seconds": 5,
    "server": {
        "host": "0.0.0.0",
        "port": 10061,
    },
    "slides_dir": "slides",
    "fullscreen": True,
    "screen_index": 1,
}

CONFIG_FILENAME = "projection_config.json"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_config_path() -> Path:
    if getattr(__import__("sys"), "frozen", False):
        base = Path(__import__("sys").executable).resolve().parent
    else:
        base = Path(__file__).resolve().parent.parent
    return base / CONFIG_FILENAME


def load_projection_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or _resolve_config_path()
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        return _deep_merge(deepcopy(DEFAULT_PROJECTION_CONFIG), raw)
    return deepcopy(DEFAULT_PROJECTION_CONFIG)


def _numeric_sort_key(p: Path) -> tuple[int, str]:
    stem = p.stem
    try:
        return (0, "")
    except ValueError:
        return (1, stem)

def _extract_number(name: str) -> int:
    import re
    m = re.search(r"\d+", name)
    return int(m.group()) if m else 0

def validate_slides_dir(slides_dir: Path) -> list[Path]:
    if not slides_dir.is_dir():
        return []
    paths = [
        p for p in slides_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
    ]
    paths.sort(key=lambda p: _extract_number(p.stem))
    return paths
