"""Slogan display, rotation management, and text tuning resolution."""

import time
from typing import Any

from app_state import APP_STATE
from config_manager import deep_merge, normalize_mojibake_text, save_config


def _normalize_slogan_lookup_key(text: str) -> str:
    """Collapse whitespace for text_tuning.by_slogan matching (supports multiline content)."""
    t = normalize_mojibake_text(text or "")
    return " ".join(t.split())


def slogan_display_text(entry: Any) -> str:
    """One-line slogan for API / UI (spaces instead of newlines)."""
    if isinstance(entry, str):
        return str(entry).strip()
    if isinstance(entry, dict):
        raw = str(entry.get("content", "") or "").strip()
        return _normalize_slogan_lookup_key(raw)
    return ""


def slogan_draw_text(entry: Any) -> str:
    """Raw slogan for rendering (preserves line breaks from config)."""
    if isinstance(entry, str):
        return str(entry).strip()
    if isinstance(entry, dict):
        return str(entry.get("content", "") or "").strip()
    return ""


def slogan_explicit_lines(draw_text: str) -> list[str] | None:
    if "\n" not in (draw_text or ""):
        return None
    lines = [ln.strip() for ln in draw_text.split("\n") if ln.strip()]
    return lines or None


def slogan_row_from_entry(entry: Any) -> int:
    """Declared row from rotation entry, else infer from explicit newlines in content."""
    if isinstance(entry, dict):
        try:
            if entry.get("row") is not None:
                return max(1, min(int(entry["row"]), 10))
        except (TypeError, ValueError):
            pass
        raw = str(entry.get("content", "") or "").strip()
        if raw and "\n" in raw:
            return max(1, len([ln for ln in raw.split("\n") if ln.strip()]))
    return 1


def normalize_slogan_lines_to_row_count(lines: list[str], target_rows: int) -> list[str]:
    """When content has more line breaks than target_rows, keep first segments and merge the rest."""
    cleaned = [ln.strip() for ln in lines if ln.strip()]
    if not cleaned:
        return [""]
    if len(cleaned) <= target_rows:
        return cleaned
    if target_rows <= 1:
        return [" ".join(cleaned)]
    if target_rows == 2:
        return [cleaned[0], " ".join(cleaned[1:]).strip()]
    head = cleaned[: target_rows - 1]
    head.append(" ".join(cleaned[target_rows - 1 :]).strip())
    return head


def get_rotation_snapshot() -> dict[str, Any]:
    rotation = APP_STATE["config"]["rotation"]
    slogans = rotation.get("slogans", []) or ["欢迎来到互动拍照区"]
    interval = max(int(rotation.get("interval_seconds", 30)), 1)
    start_time = int(rotation.get("rotation_start_time", int(time.time())))
    now = int(time.time())
    elapsed = max(now - start_time, 0)
    index = (elapsed // interval) % len(slogans)
    seconds_to_next = interval - (elapsed % interval)
    entry = slogans[index]
    draw_raw = slogan_draw_text(entry)
    display = slogan_display_text(entry)
    if not display:
        display = draw_raw or "欢迎来到互动拍照区"
    if not draw_raw:
        draw_raw = display
    return {
        "slogan": display,
        "slogan_content": draw_raw,
        "slogan_row": slogan_row_from_entry(entry),
        "index": index,
        "seconds_to_next": seconds_to_next,
        "rotation_start_time": start_time,
    }


def get_slogan_snapshot_by_sequence_no(sequence_no: int) -> dict[str, Any]:
    rotation = APP_STATE["config"]["rotation"]
    slogans = rotation.get("slogans", []) or ["欢迎来到互动拍照区"]
    if not 1 <= sequence_no <= len(slogans):
        raise ValueError(f"sequence_no out of range: {sequence_no} (valid: 1-{len(slogans)})")
    index = sequence_no - 1
    entry = slogans[index]
    draw_raw = slogan_draw_text(entry)
    display = slogan_display_text(entry)
    if not display:
        display = draw_raw or "欢迎来到互动拍照区"
    if not draw_raw:
        draw_raw = display
    return {
        "slogan": display,
        "slogan_content": draw_raw,
        "slogan_row": slogan_row_from_entry(entry),
        "index": index,
        "sequence_no": sequence_no,
        "seconds_to_next": 0,
        "rotation_start_time": 0,
    }


def set_rotation_to_index(target_index: int) -> dict[str, Any]:
    rotation = APP_STATE["config"]["rotation"]
    slogans = rotation.get("slogans", []) or ["欢迎来到互动拍照区"]
    if not 0 <= target_index < len(slogans):
        raise ValueError(f"Rotation index out of range: {target_index}")
    interval = max(int(rotation.get("interval_seconds", 30)), 1)
    now = int(time.time())
    rotation["rotation_start_time"] = now - (target_index * interval)
    save_config(APP_STATE["config"])
    return get_rotation_snapshot()


def resolve_text_tuning(slogan: str) -> dict[str, Any]:
    text_tuning_cfg = APP_STATE["config"].get("text_tuning", {})
    defaults = {
        "preferred_lines": 2,
        "line_priority": [1, 2, 3],
        "max_width_ratio": 0.82,
        "auto_break_on_punctuation": True,
        "balance_weight": 1000,
        "y_offset": 0,
        "font_scale": 1.0,
    }
    base = deep_merge(defaults, text_tuning_cfg.get("defaults", {}))
    by_slogan = text_tuning_cfg.get("by_slogan", {})
    if isinstance(by_slogan, dict):
        lookup = _normalize_slogan_lookup_key(slogan)
        per = by_slogan.get(lookup, {})
        if not per and lookup != slogan:
            per = by_slogan.get(slogan, {})
        if isinstance(per, dict):
            base = deep_merge(base, per)

    base["preferred_lines"] = max(int(base.get("preferred_lines", 2)), 1)
    max_width_ratio = float(base.get("max_width_ratio", 0.82))
    base["max_width_ratio"] = min(max(max_width_ratio, 0.5), 0.95)
    base["auto_break_on_punctuation"] = bool(base.get("auto_break_on_punctuation", True))
    base["balance_weight"] = max(int(base.get("balance_weight", 1000)), 0)
    base["y_offset"] = int(base.get("y_offset", 0))
    base["font_scale"] = max(float(base.get("font_scale", 1.0)), 0.6)
    raw_priority = base.get("line_priority", [1, 2, 3])
    resolved_priority: list[int] = []
    if isinstance(raw_priority, list):
        for item in raw_priority:
            try:
                value = int(item)
            except (TypeError, ValueError):
                continue
            if value >= 1 and value not in resolved_priority:
                resolved_priority.append(value)
    if not resolved_priority:
        resolved_priority = [1, 2, 3]
    base["line_priority"] = resolved_priority
    if "forced_lines" in base:
        base.pop("forced_lines", None)
    return base
