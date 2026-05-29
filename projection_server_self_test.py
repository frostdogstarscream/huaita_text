"""Frozen package self-test for the projection server EXE."""

from __future__ import annotations

import importlib
import traceback
from typing import Callable


def _check_import(module_name: str) -> None:
    importlib.import_module(module_name)


def _check_pyside6_widgets() -> None:
    from PySide6 import QtCore, QtGui, QtWidgets  # noqa: F401


def _check_projection_modules() -> None:
    from projection_server import projection_app_state  # noqa: F401
    from projection_server import projection_config  # noqa: F401
    from projection_server import projection_gui_app  # noqa: F401
    from projection_server import projection_main  # noqa: F401

    state = projection_app_state.get_subtitle_state()
    if not isinstance(state, dict):
        raise RuntimeError("get_subtitle_state() did not return a dict")


def run_self_test() -> int:
    checks: list[tuple[str, Callable[[], None]]] = [
        ("fastapi", lambda: _check_import("fastapi")),
        ("uvicorn", lambda: _check_import("uvicorn")),
        ("pyside6-widgets", _check_pyside6_widgets),
        ("projection-modules", _check_projection_modules),
    ]

    failed = False
    for name, check in checks:
        try:
            check()
            print(f"[OK] {name}")
        except Exception as exc:
            failed = True
            print(f"[FAIL] {name}: {exc}")
            traceback.print_exc()

    if failed:
        print("Projection server self-test failed.")
        return 1

    print("Projection server self-test passed.")
    return 0
