"""Shared Windows taskbar / window icon helpers for PySide6 GUI apps."""

from __future__ import annotations

import sys
from pathlib import Path


def _project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def resolve_icon_path() -> Path | None:
    root = _project_root()
    for candidate in (
        root / "assets" / "app_icon.ico",
        root / "_internal" / "assets" / "app_icon.ico",
    ):
        if candidate.is_file():
            return candidate
    return None


def apply_taskbar_icon(app, window=None, *, app_id: str) -> None:
    """Set AppUserModelID and window icon so Windows shows the correct taskbar icon."""
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
        except (AttributeError, OSError):
            pass

    icon_path = resolve_icon_path()
    if icon_path is None:
        return

    from PySide6.QtGui import QIcon

    icon = QIcon(str(icon_path))
    if icon.isNull():
        return

    app.setWindowIcon(icon)
    if window is not None:
        window.setWindowIcon(icon)
