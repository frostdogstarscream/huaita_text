"""Entry point for the projection server GUI + embedded FastAPI service."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path


def _write_fatal_error(exc: BaseException) -> None:
    try:
        base_dir = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
        log_path = base_dir / "projection_server_error.log"
        log_path.write_text("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)), encoding="utf-8")
    except OSError:
        pass


if __name__ == "__main__":
    try:
        if "--self-test" in sys.argv:
            from projection_server_self_test import run_self_test

            raise SystemExit(run_self_test())

        from projection_server.projection_gui_app import main

        raise SystemExit(main())
    except Exception as exc:
        _write_fatal_error(exc)
        raise
