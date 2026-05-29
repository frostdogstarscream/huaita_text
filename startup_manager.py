"""Windows startup management for the Huaita kiosk app."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runtime_paths import get_app_paths


DEFAULT_TASK_NAME = "HuaitaTextKiosk"


class AutostartError(RuntimeError):
    """Raised when a Windows autostart operation fails."""


@dataclass(frozen=True)
class AutostartStatus:
    task_name: str
    exists: bool
    returncode: int
    stdout: str = ""
    stderr: str = ""
    method: str = "task_scheduler"
    startup_entry_path: str = ""
    startup_entry_exists: bool = False
    task_exists: bool = False
    command: str = ""


def _autostart_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("autostart") or {})


def _method(config: dict[str, Any]) -> str:
    method = str(_autostart_cfg(config).get("method") or "startup_folder").strip().lower()
    return "task_scheduler" if method in {"task_scheduler", "schtasks", "scheduled_task"} else "startup_folder"


def _task_name(config: dict[str, Any]) -> str:
    name = str(_autostart_cfg(config).get("task_name") or DEFAULT_TASK_NAME).strip()
    if not name:
        raise AutostartError("autostart.task_name must not be empty")
    return name


def _startup_args(config: dict[str, Any]) -> list[str]:
    raw_args = _autostart_cfg(config).get("startup_args", [])
    if raw_args is None:
        return []
    if isinstance(raw_args, str):
        return [raw_args]
    if isinstance(raw_args, Sequence):
        return [str(value) for value in raw_args]
    return [str(raw_args)]


def _delay_seconds(config: dict[str, Any]) -> int:
    raw_delay = _autostart_cfg(config).get("delay_seconds", 10)
    try:
        return max(0, int(raw_delay))
    except (TypeError, ValueError):
        return 10


def _delay_value(config: dict[str, Any]) -> str:
    minutes, seconds = divmod(_delay_seconds(config), 60)
    return f"{minutes:04d}:{seconds:02d}"


def _run_level(config: dict[str, Any]) -> str:
    level = str(_autostart_cfg(config).get("run_level") or "LIMITED").upper()
    return "HIGHEST" if level == "HIGHEST" else "LIMITED"


def _startup_argv(config: dict[str, Any]) -> tuple[Path, list[str]]:
    app_paths = get_app_paths()
    workdir = app_paths["base_dir"]
    if getattr(sys, "frozen", False):
        argv = [str(Path(sys.executable).resolve()), *_startup_args(config)]
    else:
        argv = [str(Path(sys.executable).resolve()), str((workdir / "gui_app.py").resolve()), *_startup_args(config)]
    return workdir, argv


def build_startup_command(config: dict[str, Any]) -> str:
    """Build the command stored in Task Scheduler's /TR field."""

    workdir, argv = _startup_argv(config)
    executable = subprocess.list2cmdline(argv)
    return f'cmd.exe /C cd /d "{workdir}" && {executable}'


def _startup_folder() -> Path:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        raise AutostartError("APPDATA is required to create a current-user startup entry")
    return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"


def _startup_entry_path(config: dict[str, Any]) -> Path:
    filename = re.sub(r'[\\/:*?"<>|]+', "_", _task_name(config)).strip(". ")
    if not filename:
        filename = DEFAULT_TASK_NAME
    return _startup_folder() / f"{filename}.bat"


def _startup_script(config: dict[str, Any]) -> str:
    workdir, argv = _startup_argv(config)
    delay = _delay_seconds(config)
    executable = subprocess.list2cmdline(argv)
    lines = [
        "@echo off",
        "setlocal",
        f'cd /d "{workdir}"',
    ]
    if delay > 0:
        lines.append(f"timeout /t {delay} /nobreak >nul")
    lines.extend(
        [
            f'start "" {executable}',
            "endlocal",
            "",
        ]
    )
    return "\r\n".join(lines)


def _run_schtasks(args: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["schtasks.exe", *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise AutostartError("schtasks.exe is required on Windows to manage scheduled-task autostart") from exc


def _query_task(config: dict[str, Any]) -> AutostartStatus:
    task_name = _task_name(config)
    result = _run_schtasks(["/Query", "/TN", task_name])
    return AutostartStatus(
        task_name=task_name,
        exists=result.returncode == 0,
        returncode=result.returncode,
        stdout=result.stdout or "",
        stderr=result.stderr or "",
        method="task_scheduler",
        task_exists=result.returncode == 0,
        command=build_startup_command(config),
    )


def get_autostart_status(config: dict[str, Any]) -> AutostartStatus:
    task_name = _task_name(config)
    method = _method(config)
    startup_path = _startup_entry_path(config)
    startup_exists = startup_path.exists()
    try:
        task_status = _query_task(config)
        task_exists = task_status.task_exists
        returncode = task_status.returncode
        stdout = task_status.stdout
        stderr = task_status.stderr
    except AutostartError as exc:
        task_exists = False
        returncode = 1
        stdout = ""
        stderr = str(exc)

    exists = task_exists if method == "task_scheduler" else startup_exists
    return AutostartStatus(
        task_name=task_name,
        exists=exists,
        returncode=returncode if method == "task_scheduler" or task_exists else 0,
        stdout=stdout,
        stderr=stderr,
        method=method,
        startup_entry_path=str(startup_path),
        startup_entry_exists=startup_exists,
        task_exists=task_exists,
        command=build_startup_command(config),
    )


def _install_task_scheduler(config: dict[str, Any]) -> AutostartStatus:
    task_name = _task_name(config)
    args = [
        "/Create",
        "/F",
        "/SC",
        "ONLOGON",
        "/TN",
        task_name,
        "/TR",
        build_startup_command(config),
        "/DELAY",
        _delay_value(config),
        "/RL",
        _run_level(config),
    ]
    result = _run_schtasks(args)
    if result.returncode != 0:
        raise AutostartError((result.stderr or result.stdout or "Failed to create autostart task").strip())
    return get_autostart_status(config)


def _delete_task_if_present(config: dict[str, Any], *, raise_on_failure: bool) -> None:
    try:
        if not _query_task(config).exists:
            return
        result = _run_schtasks(["/Delete", "/F", "/TN", _task_name(config)])
        if result.returncode != 0 and raise_on_failure:
            raise AutostartError((result.stderr or result.stdout or "Failed to delete autostart task").strip())
    except AutostartError:
        if raise_on_failure:
            raise


def _install_startup_folder(config: dict[str, Any]) -> AutostartStatus:
    _delete_task_if_present(config, raise_on_failure=False)
    startup_path = _startup_entry_path(config)
    startup_path.parent.mkdir(parents=True, exist_ok=True)
    startup_path.write_text(_startup_script(config), encoding="ascii")
    return get_autostart_status(config)


def install_autostart(config: dict[str, Any]) -> AutostartStatus:
    if _method(config) == "task_scheduler":
        return _install_task_scheduler(config)
    return _install_startup_folder(config)


def uninstall_autostart(config: dict[str, Any]) -> AutostartStatus:
    startup_path = _startup_entry_path(config)
    if startup_path.exists():
        startup_path.unlink()
    _delete_task_if_present(config, raise_on_failure=True)
    return get_autostart_status(config)


def apply_autostart(config: dict[str, Any]) -> AutostartStatus:
    if bool(_autostart_cfg(config).get("enabled", False)):
        return install_autostart(config)
    return uninstall_autostart(config)
