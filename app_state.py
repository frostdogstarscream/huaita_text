"""Shared application state, path constants, and directory initialization."""

import shutil
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from config_manager import load_config, save_config
from runtime_paths import get_app_paths

APP_PATHS = get_app_paths()
BASE_DIR = APP_PATHS["base_dir"]
RESOURCE_DIR = APP_PATHS["resource_dir"]
FRONTEND_DIR = APP_PATHS["frontend_dir"]
OUTPUT_DIR = APP_PATHS["output_dir"]
CAPTURE_DIR = APP_PATHS["capture_dir"]
CUTOUT_DIR = APP_PATHS["cutout_dir"]
FINAL_DIR = APP_PATHS["final_dir"]
CONFIG_PATH = APP_PATHS["config_path"]
CAMERA_PAGE_HEARTBEAT_TIMEOUT_SECONDS = 3.0
KIOSK_PARCHMENT_BG_NAME = "kiosk-parchment-portrait-bg.png"
KIOSK_PARCHMENT_LANDSCAPE_BG_NAME = "kiosk-parchment-landscape-bg.png"

APP_STATE: dict[str, Any] = {
    "config": load_config(),
    "camera_driver": None,
    "laser_driver": None,
    "matting_service": None,
    "tasks": {},
    "tasks_lock": threading.Lock(),
    "latest_task_id": None,
    "capture_busy": False,
    "capture_busy_lock": threading.Lock(),
    "laser_trigger_running": False,
    "laser_trigger_worker": None,
    "laser_trigger_error": "",
    "camera_page_active": False,
    "camera_page_last_seen": 0.0,
    "subtitle_sync_client": None,
    "subtitle_sync_running": False,
    "subtitle_sync_worker": None,
}


def persist_laser_serial_port(serial_port: str) -> None:
    APP_STATE["config"]["laser_trigger"]["serial_port"] = serial_port
    try:
        save_config(APP_STATE["config"])
    except Exception as exc:
        APP_STATE["laser_trigger_error"] = f"Failed to persist laser serial port: {exc}"


def ensure_directories() -> None:
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    CUTOUT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)


def _write_fallback_kiosk_parchment_png(output_path: Path, width: int = 1080, height: int = 1920) -> None:
    """Warm parchment with light noise — used only if bundled PNG is missing."""
    w, h = width, height
    rng = np.random.default_rng(42)
    row = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, np.newaxis]
    col = np.linspace(0.0, 1.0, w, dtype=np.float32)[np.newaxis, :]
    noise = rng.normal(0.0, 2.9, size=(h, w)).astype(np.float32)
    r = np.clip(228.0 + row * 18.0 + col * 4.0 + noise, 0, 255)
    g = np.clip(220.0 + row * 14.0 + col * 6.0 + noise, 0, 255)
    b = np.clip(206.0 + row * 10.0 + col * 4.0 + noise, 0, 255)
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(output_path, format="PNG", optimize=True)


def ensure_kiosk_parchment_background() -> None:
    """Guarantees a readable PNG exists at frontend root (and mirrors under assets/)."""
    _ensure_single_parchment(KIOSK_PARCHMENT_BG_NAME, 1080, 1920)
    _ensure_single_parchment(KIOSK_PARCHMENT_LANDSCAPE_BG_NAME, 1920, 1080)


def _ensure_single_parchment(filename: str, width: int, height: int) -> None:
    primary = FRONTEND_DIR / filename
    assets_dir = FRONTEND_DIR / "assets"
    nested = assets_dir / filename
    min_bytes = 2048

    def usable(p: Path) -> bool:
        try:
            return p.exists() and p.stat().st_size >= min_bytes
        except OSError:
            return False

    if usable(primary):
        try:
            assets_dir.mkdir(parents=True, exist_ok=True)
            if not usable(nested):
                shutil.copy2(primary, nested)
        except OSError:
            pass
        return
    if usable(nested):
        try:
            shutil.copy2(nested, primary)
        except OSError:
            pass
        return

    assets_dir.mkdir(parents=True, exist_ok=True)
    try:
        _write_fallback_kiosk_parchment_png(primary, width, height)
        shutil.copy2(primary, nested)
    except OSError:
        pass
