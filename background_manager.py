"""Background item resolution, person/text layout merging, and background selection."""

import time
from typing import Any
from urllib.parse import quote

from app_state import APP_STATE
from config_manager import deep_merge


def get_background_items() -> list[dict[str, Any]]:
    items = APP_STATE["config"]["background_set"].get("items", [])
    resolved: list[dict[str, Any]] = []
    for order, item in enumerate(items, start=1):
        relative_path = str(item["path"]).replace("\\", "/")
        if relative_path.startswith("html-page/"):
            web_path = "/static/" + quote(relative_path[len("html-page/"):])
        else:
            web_path = "/" + quote(relative_path)
        resolved.append(
            {
                "id": item["id"],
                "name": item["name"],
                "path": relative_path,
                "preview_url": web_path,
                "order": order,
                "orientation": item.get("orientation", "portrait"),
                "person_layout": item.get("person_layout", {}),
                "text_layout": item.get("text_layout", {}),
            }
        )
    return resolved


def resolve_output_size(background_item: dict[str, Any]) -> tuple[int, int]:
    """Return (width, height) for this background's orientation."""
    cfg = APP_STATE["config"]
    output_cfg = cfg.get("output", {})
    orientation = background_item.get("orientation", "portrait")
    orientations = output_cfg.get("orientations", {})
    if orientation in orientations:
        entry = orientations[orientation]
        return (int(entry["width"]), int(entry["height"]))
    return (int(output_cfg.get("width", 1080)), int(output_cfg.get("height", 1920)))


def resolve_person_layout(background_item: dict[str, Any]) -> dict[str, Any]:
    cfg = APP_STATE["config"]
    global_layout = cfg.get("person_layout", {})
    background_layout = background_item.get("person_layout") or {}
    if not isinstance(background_layout, dict):
        background_layout = {}
    return deep_merge(global_layout, background_layout)


def resolve_text_layout(background_item: dict[str, Any]) -> dict[str, Any]:
    cfg = APP_STATE["config"]
    text_cfg = cfg.get("text_style", {})
    compose_cfg = cfg.get("compose", {})
    base_font = max(int(text_cfg.get("font_size", 72)), 1)
    base_spacing = max(int(text_cfg.get("line_spacing", 14)), 0)
    defaults = {
        "top_overlay_height": int(compose_cfg.get("top_overlay_height", 340)),
        "max_lines": 3,
        "preferred_lines": 2,
        "font_size_min": max(int(base_font * 0.58), 20),
        "font_size_max": base_font,
        "line_spacing_min": max(base_spacing // 2, 4),
        "line_spacing_max": base_spacing,
    }
    background_layout = background_item.get("text_layout") or {}
    if not isinstance(background_layout, dict):
        background_layout = {}
    merged = deep_merge(defaults, background_layout)
    merged["max_lines"] = max(int(merged.get("max_lines", 3)), 1)
    merged["preferred_lines"] = max(int(merged.get("preferred_lines", defaults["preferred_lines"])), 1)
    merged["preferred_lines"] = min(merged["preferred_lines"], merged["max_lines"])
    merged["font_size_min"] = max(int(merged.get("font_size_min", defaults["font_size_min"])), 1)
    merged["font_size_max"] = max(int(merged.get("font_size_max", defaults["font_size_max"])), merged["font_size_min"])
    merged["line_spacing_min"] = max(int(merged.get("line_spacing_min", defaults["line_spacing_min"])), 0)
    merged["line_spacing_max"] = max(int(merged.get("line_spacing_max", defaults["line_spacing_max"])), merged["line_spacing_min"])
    merged["top_overlay_height"] = max(int(merged.get("top_overlay_height", defaults["top_overlay_height"])), 80)
    return merged


def resolve_slogan_bounds(
    image_size: tuple[int, int],
    text_layout: dict[str, Any],
) -> tuple[int, int, int, int] | None:
    """Invisible layout rectangle (left, top, width, height) in pixels; None = legacy full-width band."""
    region = text_layout.get("text_region")
    if not isinstance(region, dict):
        return None
    iw, ih = image_size
    try:
        w_ratio = float(region.get("width_ratio", 0))
        h_ratio = float(region.get("height_ratio", 0))
        mt = float(region.get("margin_top_ratio", 0))
    except (TypeError, ValueError):
        return None
    if w_ratio <= 0 or h_ratio <= 0:
        return None
    rw = max(int(iw * w_ratio), 1)
    rh = max(int(ih * h_ratio), 1)
    left = max((iw - rw) // 2, 0)
    top = max(int(ih * mt), 0)
    try:
        pad = max(int(region.get("inner_padding_px", 0)), 0)
    except (TypeError, ValueError):
        pad = 0
    left = min(left + pad, iw - 1)
    top = min(top + pad, ih - 1)
    rw = max(min(rw - 2 * pad, iw - left), 1)
    rh = max(min(rh - 2 * pad, ih - top), 1)
    return (left, top, rw, rh)


def select_rotating_background(backgrounds: list[dict[str, Any]]) -> dict[str, Any]:
    if not backgrounds:
        raise ValueError("No background templates configured.")
    ui_cfg = APP_STATE["config"].get("ui", {})
    rotate_seconds = max(int(ui_cfg.get("select_background_rotate_seconds", 10)), 3)
    tick = int(time.time()) // rotate_seconds
    return backgrounds[tick % len(backgrounds)]
