"""Tests for the Qt GUI launcher helpers."""

from unittest.mock import patch

import pytest

import gui_app
from gui_app import DEFAULT_START_PATH, main, parse_args, resolve_start_url, select_runtime_port, wait_until_ready


def test_parse_args_defaults_to_kiosk_fullscreen():
    options = parse_args([])

    assert options.url == DEFAULT_START_PATH
    assert options.windowed is False
    assert options.ready_timeout_seconds >= 1
    assert options.autostart_action is None


def test_parse_args_accepts_windowed_and_custom_url():
    options = parse_args(["--windowed", "--url", "http://127.0.0.1:10051/view.html"])

    assert options.windowed is True
    assert options.url == "http://127.0.0.1:10051/view.html"


def test_resolve_start_url_builds_local_url_for_paths():
    assert resolve_start_url("/kiosk-wait.html", "0.0.0.0", 10051) == "http://127.0.0.1:10051/kiosk-wait.html"
    assert resolve_start_url("camera.html", "127.0.0.1", 10051) == "http://127.0.0.1:10051/camera.html"


def test_resolve_start_url_keeps_absolute_url():
    url = "http://127.0.0.1:10051/kiosk-wait.html"

    assert resolve_start_url(url, "127.0.0.1", 10051) == url


def test_select_runtime_port_uses_configured_port_when_available(monkeypatch):
    monkeypatch.setattr(gui_app, "is_port_available", lambda host, port: port == 10051)

    assert select_runtime_port("127.0.0.1", 10051, {"auto_port_fallback": True}) == 10051


def test_select_runtime_port_falls_back_to_next_available_port(monkeypatch):
    monkeypatch.setattr(gui_app, "is_port_available", lambda host, port: port == 10053)

    assert select_runtime_port("127.0.0.1", 10051, {"auto_port_fallback": True, "port_fallback_attempts": 5}) == 10053


def test_select_runtime_port_raises_when_fallback_disabled(monkeypatch):
    monkeypatch.setattr(gui_app, "is_port_available", lambda host, port: False)

    with pytest.raises(RuntimeError, match="already in use"):
        select_runtime_port("127.0.0.1", 10051, {"auto_port_fallback": False})


def test_wait_until_ready_raises_server_startup_error():
    class FailedServer:
        def startup_error(self):
            return RuntimeError("bind failed")

        def stopped(self):
            return True

    with pytest.raises(RuntimeError, match="bind failed"):
        wait_until_ready("http://127.0.0.1:1/", 1, FailedServer())


def test_main_autostart_status_does_not_start_gui(capsys):
    with (
        patch("gui_app.run_gui") as run_gui,
        patch("config_manager.load_config", return_value={"autostart": {"task_name": "HuaitaTextKiosk"}}),
        patch("startup_manager.get_autostart_status") as get_status,
    ):
        from startup_manager import AutostartStatus

        get_status.return_value = AutostartStatus("HuaitaTextKiosk", True, 0)

        exit_code = main(["--autostart", "status"])

    assert exit_code == 0
    run_gui.assert_not_called()
    assert "HuaitaTextKiosk" in capsys.readouterr().out
