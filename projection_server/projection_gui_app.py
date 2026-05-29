from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow


@dataclass
class GuiOptions:
    windowed: bool = False
    screen_index: int = 1


def parse_args() -> GuiOptions:
    parser = argparse.ArgumentParser(description="Projection Server GUI")
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument("--screen-index", type=int, default=1)
    args = parser.parse_args()
    return GuiOptions(windowed=args.windowed, screen_index=args.screen_index)


def _select_port(preferred: int, fallback_attempts: int = 5) -> int:
    for offset in range(fallback_attempts + 1):
        port = preferred + offset
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    return preferred


class EmbeddedServer:
    def __init__(self) -> None:
        self._server = None
        self._thread = None

    def start(self, host: str = "0.0.0.0", port: int = 10061) -> tuple[str, int]:
        from projection_server.projection_main import app
        import uvicorn

        actual_port = _select_port(port)
        # Frozen windowed EXE has no stdout/stderr; uvicorn default logging crashes on isatty().
        config = uvicorn.Config(app, host=host, port=actual_port, reload=False, log_config=None)
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True, name="projection-uvicorn")
        self._thread.start()
        return host, actual_port

    def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
        if self._thread:
            self._thread.join(timeout=5)

    def startup_error(self) -> str | None:
        if self._server and hasattr(self._server, "startup_error"):
            return self._server.startup_error()
        return None


def wait_until_ready(url: str, timeout: float = 15.0, interval: float = 0.5, server: EmbeddedServer | None = None) -> None:
    deadline = time.monotonic() + timeout
    last_err = None
    while time.monotonic() < deadline:
        if server:
            err = server.startup_error()
            if err:
                raise RuntimeError(f"Server startup error: {err}")
        try:
            urlopen(url, timeout=2)
            return
        except Exception as exc:
            last_err = exc
            time.sleep(interval)
    raise TimeoutError(f"Server not ready after {timeout}s: {last_err}")


class ProjectionWindow(QMainWindow):
    def __init__(self, options: GuiOptions) -> None:
        super().__init__()
        self._options = options
        self._last_index = -1

        self._label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet("background-color: black;")
        self.setCentralWidget(self._label)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_slide)
        self._timer.start(500)

        self._setup_shortcuts()
        self._refresh_slide()

    def _setup_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, self._toggle_play)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, self._next_slide)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, self._prev_slide)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, self.close)

    def _toggle_play(self) -> None:
        from projection_server.projection_app_state import toggle_playing
        toggle_playing()

    def _next_slide(self) -> None:
        from projection_server.projection_app_state import advance_slide
        advance_slide()

    def _prev_slide(self) -> None:
        from projection_server.projection_app_state import set_slide, PROJECTION_STATE
        idx = PROJECTION_STATE["current_index"] - 1
        set_slide(idx if idx >= 0 else PROJECTION_STATE["slide_count"] - 1)

    def _refresh_slide(self) -> None:
        from projection_server.projection_app_state import PROJECTION_STATE
        current = PROJECTION_STATE["current_index"]
        if current == self._last_index:
            return
        self._last_index = current
        paths = PROJECTION_STATE["slide_paths"]
        if 0 <= current < len(paths):
            pixmap = QPixmap(str(paths[current]))
            if not pixmap.isNull():
                scaled = pixmap.scaled(
                    self._label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self._label.setPixmap(scaled)
                return
        self._label.clear()


def run_gui(options: GuiOptions) -> None:
    from projection_server.projection_app_state import PROJECTION_STATE

    server = EmbeddedServer()
    cfg = PROJECTION_STATE["config"]["server"]
    host, port = server.start(cfg.get("host", "0.0.0.0"), cfg.get("port", 10061))

    try:
        wait_until_ready(f"http://127.0.0.1:{port}/api/health", server=server)
    except Exception as exc:
        print(f"[ProjectionGUI] Server failed to start: {exc}")
        server.stop()
        return

    app = QApplication(sys.argv[:1])
    from gui_icon import apply_taskbar_icon

    screens = app.screens()
    target_screen = screens[options.screen_index] if options.screen_index < len(screens) else screens[0]

    window = ProjectionWindow(options)
    apply_taskbar_icon(app, window, app_id="HuaitaText.ProjectionServer")
    window.setWindowTitle("Projection Server")

    if options.windowed:
        window.resize(960, 540)
        window.show()
    else:
        window.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        geo = target_screen.geometry()
        window.setGeometry(geo)
        window.showFullScreen()

    app.aboutToQuit.connect(server.stop)
    app.exec()
    server.stop()


def main() -> None:
    options = parse_args()
    run_gui(options)


if __name__ == "__main__":
    main()
