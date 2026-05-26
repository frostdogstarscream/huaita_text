"""Tests for camera driver startup behavior."""

import time
from unittest.mock import MagicMock, patch

import numpy as np

from camera_driver import CameraDriver


def _camera_config():
    return {
        "selection_mode": "fixed",
        "index": 0,
        "backend": "CAP_ANY",
        "probe_indices": [0],
        "preferred_indices": [0],
        "backend_order": ["CAP_ANY"],
        "width": 1280,
        "height": 720,
        "fps": 20,
        "auto_focus": False,
        "jpeg_quality": 90,
        "log_enabled": False,
        "stale_frame_seconds": 5.0,
    }


def test_start_schedules_capture_probe_without_blocking():
    driver = CameraDriver(_camera_config())
    loop = MagicMock()
    thread = MagicMock()

    with (
        patch("camera_driver.asyncio.get_running_loop", return_value=loop),
        patch("camera_driver.threading.Thread", return_value=thread) as thread_cls,
        patch.object(driver, "_open_capture") as open_capture,
    ):
        driver.start()

    assert driver.running is True
    open_capture.assert_not_called()
    thread_cls.assert_called_once_with(target=driver._init_and_start_reading, name="huaita-camera-start", daemon=True)
    thread.start.assert_called_once()
    assert driver._startup_thread is thread
    loop.call_later.assert_not_called()


def test_camera_log_writes_to_configured_path(tmp_path):
    log_path = tmp_path / "camera.log"
    config = _camera_config() | {"log_enabled": True, "log_path": str(log_path)}
    driver = CameraDriver(config)

    driver._log("probe message")

    text = log_path.read_text(encoding="utf-8")
    assert "camera driver initialized" in text
    assert "probe message" in text
    status = driver.status()
    assert status["log_enabled"] is True
    assert status["log_path"] == str(log_path)
    assert status["read_fail_count"] == 0


def test_open_capture_logs_failed_probe(tmp_path):
    log_path = tmp_path / "camera.log"
    config = _camera_config() | {"log_enabled": True, "log_path": str(log_path)}
    driver = CameraDriver(config)
    driver._probe_camera = MagicMock(return_value={
        "index": 0,
        "backend": "CAP_ANY",
        "opened": False,
        "read_ok": False,
        "shape": None,
        "error": "open failed for test",
        "capture": None,
        "frame": None,
    })

    driver._open_capture()

    text = log_path.read_text(encoding="utf-8")
    assert "open capture attempt #1" in text
    assert "probing camera index=0 backend=CAP_ANY" in text
    assert "open capture failed" in text
    assert driver.last_error == "failed to find usable camera; last error: open failed for test"


class _FakeCapture:
    def __init__(self, *, opened=True, retrieve_ok=False):
        self.opened = opened
        self.retrieve_ok = retrieve_ok
        self.released = False

    def isOpened(self):
        return self.opened and not self.released

    def grab(self):
        return True

    def retrieve(self):
        if self.retrieve_ok:
            return True, np.zeros((2, 2, 3), dtype=np.uint8)
        return False, None

    def release(self):
        self.released = True


def test_read_frame_logs_failure_and_releases_capture(tmp_path):
    log_path = tmp_path / "camera.log"
    config = _camera_config() | {"log_enabled": True, "log_path": str(log_path)}
    driver = CameraDriver(config)
    capture = _FakeCapture(retrieve_ok=False)
    driver.running = True
    driver.capture = capture
    driver._schedule_next_read = MagicMock()

    driver._read_frame()

    assert driver.read_fail_count == 1
    assert driver.capture is None
    assert capture.released is True
    assert "camera read failed count=1" in log_path.read_text(encoding="utf-8")


def test_stale_frame_watchdog_releases_capture(tmp_path):
    log_path = tmp_path / "camera.log"
    config = _camera_config() | {
        "log_enabled": True,
        "log_path": str(log_path),
        "stale_frame_seconds": 0.1,
    }
    driver = CameraDriver(config)
    capture = _FakeCapture(retrieve_ok=True)
    driver.running = True
    driver.capture = capture
    driver.last_frame_time = time.time() - 1.0

    driver._restart_if_frame_stale()

    assert driver.capture is None
    assert capture.released is True
    assert "camera frame stale" in log_path.read_text(encoding="utf-8")
