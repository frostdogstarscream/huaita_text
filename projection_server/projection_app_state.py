from __future__ import annotations

import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from projection_server.projection_config import load_projection_config, validate_slides_dir

_BEIJING_TZ = timezone(timedelta(hours=8))


def _now_iso() -> str:
    return datetime.now(_BEIJING_TZ).isoformat(timespec="seconds")


_config = load_projection_config()
_slides_dir = Path(_config["slides_dir"])
if not _slides_dir.is_absolute():
    if getattr(__import__("sys"), "frozen", False):
        _base = Path(__import__("sys").executable).resolve().parent
    else:
        _base = Path(__file__).resolve().parent.parent
    _slides_dir = _base / _slides_dir

_slide_paths = validate_slides_dir(_slides_dir)

PROJECTION_STATE: dict[str, Any] = {
    "config": _config,
    "slide_paths": _slide_paths,
    "slide_count": len(_slide_paths),
    "current_index": 0,
    "playing": True,
    "revision": 0,
    "changed_at": _now_iso(),
    "state_lock": threading.Lock(),
    "playlist_valid": len(_slide_paths) > 0,
    "running": False,
    "rotation_worker": None,
}


def advance_slide() -> None:
    with PROJECTION_STATE["state_lock"]:
        if not PROJECTION_STATE["playlist_valid"]:
            return
        PROJECTION_STATE["current_index"] = (
            PROJECTION_STATE["current_index"] + 1
        ) % PROJECTION_STATE["slide_count"]
        PROJECTION_STATE["revision"] += 1
        PROJECTION_STATE["changed_at"] = _now_iso()


def set_slide(index: int) -> None:
    with PROJECTION_STATE["state_lock"]:
        if not PROJECTION_STATE["playlist_valid"]:
            return
        if 0 <= index < PROJECTION_STATE["slide_count"]:
            PROJECTION_STATE["current_index"] = index
            PROJECTION_STATE["revision"] += 1
            PROJECTION_STATE["changed_at"] = _now_iso()


def toggle_playing() -> bool:
    with PROJECTION_STATE["state_lock"]:
        PROJECTION_STATE["playing"] = not PROJECTION_STATE["playing"]
        return PROJECTION_STATE["playing"]


def get_subtitle_state() -> dict[str, Any]:
    with PROJECTION_STATE["state_lock"]:
        cfg = PROJECTION_STATE["config"]
        seq = PROJECTION_STATE["current_index"] + 1
        return {
            "ok": PROJECTION_STATE["playlist_valid"],
            "playlist_id": cfg.get("playlist_id", ""),
            "slide_count": PROJECTION_STATE["slide_count"],
            "sequence_no": seq,
            "slide_id": f"slide_{seq:03d}",
            "interval_seconds": cfg.get("interval_seconds", 5),
            "playing": PROJECTION_STATE["playing"],
            "revision": PROJECTION_STATE["revision"],
            "changed_at": PROJECTION_STATE["changed_at"],
        }
