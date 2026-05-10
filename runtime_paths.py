from __future__ import annotations

import sys
from pathlib import Path


def get_runtime_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def get_resource_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def get_app_paths() -> dict[str, Path]:
    base_dir = get_runtime_base_dir()
    resource_dir = get_resource_base_dir()
    output_dir = base_dir / "generated"
    return {
        "base_dir": base_dir,
        "resource_dir": resource_dir,
        "frontend_dir": resource_dir / "html-page",
        "config_path": base_dir / "config.json",
        "output_dir": output_dir,
        "capture_dir": output_dir / "captures",
        "cutout_dir": output_dir / "cutouts",
        "final_dir": output_dir / "final",
        "fonts_dir": resource_dir / "fonts",
    }
