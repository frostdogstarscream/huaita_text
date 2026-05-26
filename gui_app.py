"""Qt fullscreen GUI shell for the Huaita kiosk web experience."""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen


DEFAULT_START_PATH = "/kiosk-wait.html"
DEFAULT_READY_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class GuiOptions:
    """Runtime options for the kiosk GUI shell."""

    url: str
    windowed: bool
    ready_timeout_seconds: float
    autostart_action: str | None = None


def parse_args(argv: Sequence[str] | None = None) -> GuiOptions:
    parser = argparse.ArgumentParser(description="Start the Huaita kiosk GUI.")
    parser.add_argument(
        "--url",
        default=DEFAULT_START_PATH,
        help="Initial page URL or path. Defaults to /kiosk-wait.html.",
    )
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="Run in a resizable window for development instead of fullscreen.",
    )
    parser.add_argument(
        "--ready-timeout",
        type=float,
        default=DEFAULT_READY_TIMEOUT_SECONDS,
        help="Seconds to wait for the local FastAPI service before loading the GUI.",
    )
    parser.add_argument(
        "--autostart",
        choices=("apply", "install", "uninstall", "status"),
        help="Manage the Windows Task Scheduler autostart entry and exit without opening the GUI.",
    )
    args = parser.parse_args(argv)
    return GuiOptions(
        url=str(args.url),
        windowed=bool(args.windowed),
        ready_timeout_seconds=max(float(args.ready_timeout), 1.0),
        autostart_action=args.autostart,
    )


def _localhost_for_browser(host: str) -> str:
    if host in ("0.0.0.0", "::", ""):
        return "127.0.0.1"
    return host


def resolve_start_url(raw_url: str, host: str, port: int) -> str:
    parsed = urlparse(raw_url)
    if parsed.scheme in ("http", "https"):
        return raw_url
    path = raw_url if raw_url.startswith("/") else f"/{raw_url}"
    return f"http://{_localhost_for_browser(host)}:{port}{path}"


class EmbeddedServer:
    """Run uvicorn in a background thread so Qt can own the main thread."""

    def __init__(self) -> None:
        self._server = None
        self._thread: threading.Thread | None = None
        self._startup_error: BaseException | None = None

    def start(self) -> tuple[str, int]:
        import uvicorn
        from main import APP_STATE, app

        server_cfg = APP_STATE["config"]["server"]
        host = str(server_cfg.get("host", "127.0.0.1"))
        port = select_runtime_port(host, int(server_cfg.get("port", 10051)), server_cfg)

        config = uvicorn.Config(app, host=host, port=port, reload=False, log_config=None)
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._run_server, name="huaita-uvicorn", daemon=True)
        self._thread.start()
        return host, port

    def _run_server(self) -> None:
        try:
            self._server.run()
        except BaseException as exc:
            self._startup_error = exc
            _write_fatal_error(exc)

    def startup_error(self) -> BaseException | None:
        return self._startup_error

    def stopped(self) -> bool:
        return self._thread is not None and not self._thread.is_alive()

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)


def _bind_probe_host(host: str) -> str:
    if host in ("", "0.0.0.0", "::"):
        return "127.0.0.1"
    return host


def is_port_available(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((_bind_probe_host(host), port))
        return True
    except OSError:
        return False


def select_runtime_port(host: str, configured_port: int, server_cfg: dict) -> int:
    if is_port_available(host, configured_port):
        return configured_port
    if not bool(server_cfg.get("auto_port_fallback", True)):
        raise RuntimeError(f"Configured server port is already in use: {host}:{configured_port}")

    attempts = max(1, int(server_cfg.get("port_fallback_attempts", 10)))
    for port in range(configured_port + 1, configured_port + attempts + 1):
        if is_port_available(host, port):
            return port
    raise RuntimeError(f"No available server port found from {configured_port} to {configured_port + attempts}")


def wait_until_ready(url: str, timeout_seconds: float, server: EmbeddedServer | None = None) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if server is not None:
            startup_error = server.startup_error()
            if startup_error is not None:
                raise RuntimeError(f"FastAPI service failed to start: {startup_error}") from startup_error
            if server.stopped():
                raise RuntimeError("FastAPI service stopped before it became ready")
        try:
            with urlopen(url, timeout=1) as response:
                if 200 <= response.status < 500:
                    return True
        except (OSError, URLError):
            time.sleep(0.5)
    return False


def _configure_web_engine(view) -> None:
    from PySide6.QtWebEngineCore import QWebEngineSettings

    settings = view.settings()
    settings.setAttribute(QWebEngineSettings.WebAttribute.FullScreenSupportEnabled, True)
    settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
    settings.setAttribute(QWebEngineSettings.WebAttribute.PlaybackRequiresUserGesture, False)
    settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptCanOpenWindows, False)


def run_gui(options: GuiOptions) -> int:
    try:
        from PySide6.QtCore import Qt, QUrl
        from PySide6.QtWidgets import QApplication, QMainWindow
        from PySide6.QtWebEngineWidgets import QWebEngineView
    except ImportError as exc:
        raise RuntimeError("PySide6 is required for GUI mode. Install requirements.txt first.") from exc

    embedded_server = EmbeddedServer()
    host, port = embedded_server.start()
    start_url = resolve_start_url(options.url, host, port)

    if not wait_until_ready(start_url, options.ready_timeout_seconds, embedded_server):
        embedded_server.stop()
        raise RuntimeError(f"FastAPI service did not become ready: {start_url}")

    app = QApplication(sys.argv[:1])
    app.setApplicationName("Huaita Text Kiosk")
    app.aboutToQuit.connect(embedded_server.stop)

    window = QMainWindow()
    window.setWindowTitle("Huaita Text Kiosk")
    window.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)

    view = QWebEngineView(window)
    view.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
    _configure_web_engine(view)
    view.load(QUrl(start_url))

    window.setCentralWidget(view)
    if options.windowed:
        window.resize(540, 960)
        window.show()
    else:
        window.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        window.showFullScreen()

    try:
        return app.exec()
    finally:
        embedded_server.stop()


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if "--self-test" in argv:
        from package_self_test import run_self_test

        return run_self_test()
    options = parse_args(argv)
    if options.autostart_action:
        from config_manager import load_config
        from startup_manager import (
            AutostartError,
            apply_autostart,
            get_autostart_status,
            install_autostart,
            uninstall_autostart,
        )

        config = load_config()
        try:
            if options.autostart_action == "apply":
                status = apply_autostart(config)
            elif options.autostart_action == "install":
                status = install_autostart(config)
            elif options.autostart_action == "uninstall":
                status = uninstall_autostart(config)
            else:
                status = get_autostart_status(config)
        except AutostartError as exc:
            print(f"Autostart error: {exc}", file=sys.stderr)
            return 1
        state = "installed" if status.exists else "not installed"
        print(f"Autostart task {status.task_name}: {state}")
        return 0
    return run_gui(options)


def _write_fatal_error(exc: BaseException) -> None:
    try:
        base_dir = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
        log_path = base_dir / "huaita_text_error.log"
        log_path.write_text("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)), encoding="utf-8")
    except OSError:
        pass


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        _write_fatal_error(exc)
        raise
