"""Configuration loading, merging, normalization, and persistence."""

import json
import os
import time
from pathlib import Path
from typing import Any

from runtime_paths import get_app_paths


DEFAULT_CONFIG: dict[str, Any] = {
    "server": {
        "host": "127.0.0.1",
        "port": 10051,
        "auto_port_fallback": True,
        "port_fallback_attempts": 10,
    },
    "autostart": {
        "enabled": False,
        "method": "startup_folder",
        "task_name": "HuaitaTextKiosk",
        "delay_seconds": 10,
        "run_level": "LIMITED",
        "startup_args": [],
    },
    "camera": {
        "selection_mode": "auto_prefer_external",
        "index": 0,
        "backend": "CAP_DSHOW",
        "probe_indices": [0, 1, 2, 3, 4, 5],
        "preferred_indices": [2, 1, 3, 4, 5, 0],
        "backend_order": ["CAP_ANY", "CAP_MSMF", "CAP_DSHOW"],
        "width": 1280,
        "height": 720,
        "fps": 20,
        "auto_focus": True,
        "jpeg_quality": 90,
        "log_enabled": True,
        "log_path": "",
        "stale_frame_seconds": 5.0,
    },
    "rotation": {
        "interval_seconds": 30,
        "rotation_start_time": 0,
        "slogans": [
            "欢迎来到互动拍照区",
            "定格精彩瞬间，留下专属海报",
            "四款背景一键生成，马上分享",
            "请站稳看镜头，准备拍摄",
        ],
    },
    "background_set": {
        "items": [
            {"id": "bg_001", "name": "背景一", "path": "html-page/assets/photos/1.jpg"},
            {"id": "bg_002", "name": "背景二", "path": "html-page/assets/photos/2.jpg"},
            {"id": "bg_003", "name": "背景三", "path": "html-page/assets/photos/3.jpg"},
            {"id": "bg_004", "name": "背景四", "path": "html-page/assets/photos/4.jpg"},
        ]
    },
    "output": {
        "width": 1080,
        "height": 1920,
        "jpeg_quality": 92,
    },
    "ui": {
        "kiosk_idle_return_seconds": 30,
        "select_background_rotate_seconds": 10,
    },
    "person_layout": {
        "target_height_ratio": 0.72,
        "bottom_margin": 80,
        "center_x_ratio": 0.50,
        "center_y_offset": 0,
    },
    "compose": {
        "top_overlay_height": 340,
        "overlay_opacity": 120,
    },
    "matting_api": {
        "provider": "ali_segment_body",
        "bucket": "huaita-person-img",
        "region": "cn-shanghai",
        "oss_endpoint": "https://oss-cn-shanghai.aliyuncs.com",
        "imageseg_endpoint": "imageseg.cn-shanghai.aliyuncs.com",
        "output_dir": "generated/cutouts",
        "max_image_edge": 2000,
        "use_seedream": False,
        "use_suxiaoban": False,
        "suxiaoban": {},
    },
    "text_style": {
        "font_size": 72,
        "top_margin": 88,
        "line_spacing": 14,
        "stroke_width": 3,
        "fill": "#FFFFFF",
        "stroke_fill": "#452d00",
        "style_mode": "gold_layered",
        "gold_palette": ["#fff7d8", "#f3cb63", "#b6751f"],
        "outline_dark": "#6c3f0a",
        "outline_width": 1,
        "shadow_color": "#4a2908",
        "shadow_offset": [1, 2],
        "shadow_blur": 1,
        "shadow_alpha": 96,
        "highlight_color": "#fffbe8",
        "highlight_offset": [0, -1],
        "highlight_alpha": 140,
        "specular_color": "#fffef4",
        "specular_strength": 188,
        "specular_band_top_ratio": 0.22,
        "specular_band_height_ratio": 0.18,
        "inner_glow_color": "#fff3b8",
        "inner_glow_alpha": 58,
    },
    "text_tuning": {
        "defaults": {
            "preferred_lines": 2,
            "max_width_ratio": 0.82,
            "auto_break_on_punctuation": True,
            "balance_weight": 1000,
        },
        "by_slogan": {
            "只要还有一个人活着 就要守住阵地": {
                "y_offset": -6,
                "font_scale": 1.03,
            }
        },
    },
    "laser_trigger": {
        "enabled": True,
        "serial_port": "COM3",
        "baudrate": 19200,
        "bytesize": 8,
        "stopbits": 1,
        "parity": "N",
        "timeout_seconds": 0.2,
        "measure_mode": "continuous_fast_20hz",
        "trigger_min_cm": 80,
        "trigger_max_cm": 150,
        "stable_samples": 3,
        "stable_delta_cm": 5,
        "countdown_seconds": 5,
        "burst_count": 4,
        "burst_interval_seconds": 0.2,
        "cooldown_ms": 5000,
        "require_leave_before_retrigger": True,
        "leave_min_cm": 180,
    },
}


def _config_path() -> Path:
    return get_app_paths()["config_path"]


class ConfigError(RuntimeError):
    pass


class ConfigLoadError(ConfigError):
    pass


class ConfigSaveError(ConfigError):
    pass


def normalize_mojibake_text(value: str) -> str:
    if not value:
        return value
    try:
        fixed = value.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value
    return fixed if fixed and fixed != value else value


def normalize_text_tree(value: Any) -> Any:
    if isinstance(value, str):
        return normalize_mojibake_text(value)
    if isinstance(value, list):
        return [normalize_text_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_text_tree(item) for key, item in value.items()}
    return value


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _default_config() -> dict[str, Any]:
    initial = normalize_text_tree(deep_merge(DEFAULT_CONFIG, {}))
    initial["rotation"]["rotation_start_time"] = int(time.time())
    return initial


def _write_config_json(config_path: Path, config: dict[str, Any]) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(normalize_text_tree(config), ensure_ascii=False, indent=2)
    temp_path = config_path.with_name(f"{config_path.name}.tmp")
    try:
        temp_path.write_text(payload, encoding="utf-8")
        os.replace(temp_path, config_path)
    except OSError as exc:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass
        raise ConfigSaveError(f"Failed to save config: {config_path}") from exc


def _backup_unreadable_config(config_path: Path) -> Path | None:
    if not config_path.exists():
        return None
    backup_path = config_path.with_name(f"{config_path.stem}.invalid-{int(time.time())}{config_path.suffix}")
    try:
        os.replace(config_path, backup_path)
    except OSError:
        return None
    return backup_path


def load_config() -> dict[str, Any]:
    config_path = _config_path()
    if not config_path.exists():
        initial = _default_config()
        _write_config_json(config_path, initial)
        return initial

    try:
        raw_source = json.loads(config_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        backup_path = _backup_unreadable_config(config_path)
        initial = _default_config()
        try:
            _write_config_json(config_path, initial)
        except ConfigSaveError:
            raise ConfigLoadError(f"Failed to load config and restore defaults: {config_path}") from exc
        if backup_path:
            initial["_config_warning"] = f"Invalid config was backed up to {backup_path.name}"
        return initial

    raw = normalize_text_tree(raw_source)
    config = normalize_text_tree(deep_merge(DEFAULT_CONFIG, raw))
    legacy_compose = raw.get("compose", {})
    if "person_layout" not in raw:
        if "target_height_ratio" in legacy_compose:
            config["person_layout"]["target_height_ratio"] = legacy_compose["target_height_ratio"]
        if "bottom_margin" in legacy_compose:
            config["person_layout"]["bottom_margin"] = legacy_compose["bottom_margin"]
    if not config["rotation"].get("rotation_start_time"):
        config["rotation"]["rotation_start_time"] = int(time.time())
    if raw != raw_source:
        _write_config_json(config_path, config)
    return config


def save_config(config: dict[str, Any]) -> None:
    _write_config_json(_config_path(), config)
