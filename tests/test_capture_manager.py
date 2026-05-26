"""Test capture_manager: task creation, mutex, task status, QR generation."""

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from capture_manager import (
    CaptureBusyError,
    build_qr_image,
    get_latest_task,
    is_capture_busy,
    mark_camera_page_active,
    release_capture_slot,
    start_capture_task,
    try_acquire_capture_slot,
    update_task,
)


class TestCaptureSlotMutex:
    def test_acquire_succeeds_when_free(self, patched_app_state):
        patched_app_state["capture_busy"] = False
        assert try_acquire_capture_slot() is True
        assert patched_app_state["capture_busy"] is True

    def test_acquire_fails_when_busy(self, patched_app_state):
        patched_app_state["capture_busy"] = True
        assert try_acquire_capture_slot() is False

    def test_release_clears_busy(self, patched_app_state):
        patched_app_state["capture_busy"] = True
        release_capture_slot()
        assert patched_app_state["capture_busy"] is False

    def test_is_capture_busy(self, patched_app_state):
        patched_app_state["capture_busy"] = False
        assert is_capture_busy() is False
        patched_app_state["capture_busy"] = True
        assert is_capture_busy() is True


class TestTaskManagement:
    def test_update_task_creates_new(self, patched_app_state):
        with patched_app_state["tasks_lock"]:
            patched_app_state["tasks"] = {}
        update_task("task_001", status="queued", message="测试")
        with patched_app_state["tasks_lock"]:
            assert "task_001" in patched_app_state["tasks"]
            assert patched_app_state["tasks"]["task_001"]["status"] == "queued"

    def test_update_task_merges_fields(self, patched_app_state):
        with patched_app_state["tasks_lock"]:
            patched_app_state["tasks"] = {}
        update_task("task_002", status="queued")
        update_task("task_002", message="更新消息", results=[1, 2, 3])
        with patched_app_state["tasks_lock"]:
            task = patched_app_state["tasks"]["task_002"]
            assert task["status"] == "queued"
            assert task["message"] == "更新消息"
            assert task["results"] == [1, 2, 3]

    def test_get_latest_task_returns_none_when_empty(self, patched_app_state):
        patched_app_state["latest_task_id"] = None
        assert get_latest_task() is None

    def test_get_latest_task_returns_dict(self, patched_app_state):
        with patched_app_state["tasks_lock"]:
            patched_app_state["tasks"] = {}
        patched_app_state["latest_task_id"] = "task_latest"
        update_task("task_latest", status="completed", results=[])
        task = get_latest_task()
        assert task is not None
        assert task["status"] == "completed"

    def test_start_capture_task_returns_task_dict(self, patched_app_state):
        patched_app_state["capture_busy"] = False
        task = start_capture_task(source="manual")
        assert "task_id" in task
        assert task["status"] in ("queued", "processing")  # daemon 线程可能已开始处理
        assert task["trigger_source"] == "manual"

        # Clean up: release the slot since process_capture_task runs in daemon thread
        time.sleep(0.1)

    def test_start_capture_task_fails_when_busy(self, patched_app_state):
        patched_app_state["capture_busy"] = True
        with pytest.raises(CaptureBusyError):
            start_capture_task(source="manual")


class TestCameraPageActive:
    def test_mark_active(self, patched_app_state):
        mark_camera_page_active(True)
        assert patched_app_state["camera_page_active"] is True

    def test_mark_inactive(self, patched_app_state):
        mark_camera_page_active(True)
        mark_camera_page_active(False)
        assert patched_app_state["camera_page_active"] is False

    def test_stale_heartbeat_marks_inactive(self, patched_app_state):
        from capture_manager import is_camera_page_active, CAMERA_PAGE_HEARTBEAT_TIMEOUT_SECONDS

        patched_app_state["camera_page_active"] = True
        patched_app_state["camera_page_last_seen"] = time.time() - CAMERA_PAGE_HEARTBEAT_TIMEOUT_SECONDS - 10
        assert is_camera_page_active() is False


class TestQRCode:
    def test_build_qr_returns_bytes(self):
        result = build_qr_image("https://example.com/test")
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_build_qr_valid_png(self):
        import io
        from PIL import Image as PILImage

        result = build_qr_image("test data")
        img = PILImage.open(io.BytesIO(result))
        assert img.format == "PNG"
        assert img.size == (280, 280)
