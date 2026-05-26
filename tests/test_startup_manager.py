"""Tests for Windows autostart management."""

import subprocess
import sys
from unittest.mock import patch

import startup_manager
from startup_manager import (
    apply_autostart,
    build_startup_command,
    get_autostart_status,
    install_autostart,
    uninstall_autostart,
)


def _completed(args=None, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args or ["schtasks.exe"], returncode, stdout, stderr)


def _config(**autostart):
    cfg = {
        "autostart": {
            "enabled": False,
            "method": "startup_folder",
            "task_name": "HuaitaTextKiosk",
            "delay_seconds": 10,
            "run_level": "LIMITED",
            "startup_args": [],
        }
    }
    cfg["autostart"].update(autostart)
    return cfg


def _patch_paths(tmp_path, monkeypatch):
    appdata = tmp_path / "AppData" / "Roaming"
    monkeypatch.setenv("APPDATA", str(appdata))
    monkeypatch.setattr(startup_manager, "get_app_paths", lambda: {"base_dir": tmp_path / "app"})
    return appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"


def test_build_startup_command_uses_dev_python_and_gui_app(tmp_path, monkeypatch):
    monkeypatch.setattr(startup_manager, "get_app_paths", lambda: {"base_dir": tmp_path})
    monkeypatch.setattr(sys, "frozen", False, raising=False)

    command = build_startup_command(_config(startup_args=["--url", "/camera.html"]))

    assert "cmd.exe /C cd /d" in command
    assert str(tmp_path) in command
    assert str(tmp_path / "gui_app.py") in command
    assert "--url /camera.html" in command


def test_build_startup_command_uses_frozen_executable(tmp_path, monkeypatch):
    exe_path = tmp_path / "huaita_text.exe"
    monkeypatch.setattr(startup_manager, "get_app_paths", lambda: {"base_dir": tmp_path})
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe_path))

    command = build_startup_command(_config())

    assert str(exe_path) in command
    assert "gui_app.py" not in command


def test_default_status_uses_startup_folder_method(tmp_path, monkeypatch):
    startup_dir = _patch_paths(tmp_path, monkeypatch)
    with patch("startup_manager.subprocess.run", return_value=_completed(returncode=1, stderr="missing")):
        status = get_autostart_status(_config())

    assert status.method == "startup_folder"
    assert status.exists is False
    assert status.startup_entry_exists is False
    assert status.task_exists is False
    assert status.startup_entry_path == str(startup_dir / "HuaitaTextKiosk.bat")


def test_install_autostart_creates_startup_bat_without_admin(tmp_path, monkeypatch):
    startup_dir = _patch_paths(tmp_path, monkeypatch)
    with patch("startup_manager.subprocess.run", return_value=_completed(returncode=1, stderr="missing")) as run:
        status = install_autostart(_config(delay_seconds=12))

    script_path = startup_dir / "HuaitaTextKiosk.bat"
    script = script_path.read_text(encoding="ascii")
    assert status.exists is True
    assert status.startup_entry_exists is True
    assert status.task_exists is False
    assert script_path.exists()
    assert "timeout /t 12 /nobreak >nul" in script
    assert "cd /d" in script
    assert "gui_app.py" in script
    assert run.call_count >= 1
    assert "/Create" not in run.call_args_list[0].args[0]


def test_install_startup_folder_deletes_existing_task_best_effort(tmp_path, monkeypatch):
    _patch_paths(tmp_path, monkeypatch)
    with patch(
        "startup_manager.subprocess.run",
        side_effect=[
            _completed(returncode=0),  # query old task before install
            _completed(returncode=0),  # delete old task
            _completed(returncode=1),  # status query after install
        ],
    ) as run:
        install_autostart(_config())

    delete_args = run.call_args_list[1].args[0]
    assert delete_args[:3] == ["schtasks.exe", "/Delete", "/F"]


def test_uninstall_autostart_removes_startup_bat_and_task(tmp_path, monkeypatch):
    startup_dir = _patch_paths(tmp_path, monkeypatch)
    startup_dir.mkdir(parents=True)
    script_path = startup_dir / "HuaitaTextKiosk.bat"
    script_path.write_text("@echo off\r\n", encoding="ascii")

    with patch(
        "startup_manager.subprocess.run",
        side_effect=[
            _completed(returncode=0),  # query existing task
            _completed(returncode=0),  # delete existing task
            _completed(returncode=1),  # status query
        ],
    ) as run:
        status = uninstall_autostart(_config())

    assert script_path.exists() is False
    assert status.exists is False
    assert run.call_args_list[1].args[0][:3] == ["schtasks.exe", "/Delete", "/F"]


def test_task_scheduler_method_creates_onlogon_task_with_delay_and_run_level(tmp_path, monkeypatch):
    _patch_paths(tmp_path, monkeypatch)
    with patch("startup_manager.subprocess.run", side_effect=[_completed(), _completed()]) as run:
        status = install_autostart(_config(method="task_scheduler", delay_seconds=75, run_level="HIGHEST"))

    create_args = run.call_args_list[0].args[0]
    assert create_args[:4] == ["schtasks.exe", "/Create", "/F", "/SC"]
    assert "ONLOGON" in create_args
    assert "/TN" in create_args
    assert "HuaitaTextKiosk" in create_args
    assert "/DELAY" in create_args
    assert "0001:15" in create_args
    assert "/RL" in create_args
    assert "HIGHEST" in create_args
    assert status.exists is True


def test_apply_autostart_installs_or_uninstalls_from_config():
    with patch("startup_manager.install_autostart") as install, patch("startup_manager.uninstall_autostart") as uninstall:
        apply_autostart(_config(enabled=True))
        apply_autostart(_config(enabled=False))

    install.assert_called_once()
    uninstall.assert_called_once()
