"""Test capture_manager: task creation, mutex, task status, QR generation."""

import threading
import time
from pathlib import Path
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

    def test_process_capture_task_uses_tracked_pipeline_when_enabled(self, patched_app_state):
        import capture_manager

        patched_app_state["config"]["matting_api"]["provider"] = "tracked_matanyone"
        patched_app_state["config"]["tracked_matting"] = {
            "enabled": True,
            "input_frame_count": 16,
            "output_frame_indices": [3, 7, 10, 13],
            "timeout_seconds": 20,
        }

        fake_capture_data = [
            (Path("frame1.jpg"), "task_1"),
            (Path("frame2.jpg"), "task_2"),
            (Path("frame3.jpg"), "task_3"),
            (Path("frame4.jpg"), "task_4"),
        ]

        with patch.object(capture_manager, "get_rotation_snapshot", return_value={"slogan": "test", "slogan_content": "test", "slogan_row": 1}), \
             patch.object(capture_manager, "get_background_items", return_value=[{"id": "bg_001", "name": "bg", "path": "x"}]), \
             patch.object(capture_manager, "select_rotating_background", return_value={"id": "bg_001", "name": "bg", "path": "x"}), \
             patch.object(capture_manager, "_capture_tracked_sequence_frames", return_value={"capture_data": fake_capture_data, "capture_urls": ["u1", "u2"], "frame_paths": [], "video_path": Path("v.avi"), "output_indices": [0, 1, 2, 3], "shot_task_ids": ["a", "b", "c", "d"]}) as m_capture, \
             patch.object(capture_manager, "_segment_tracked_sequence", return_value=([Image.new("RGBA", (10, 10), (0, 0, 0, 255))] * 4, ["c1", "c2", "c3", "c4"])) as m_segment, \
             patch.object(capture_manager, "_segment_captures_parallel") as m_old_segment, \
             patch.object(capture_manager, "_compose_capture_results", return_value=[{"ok": True}]) as m_compose:
            capture_manager.process_capture_task("task_x")

        assert m_capture.called
        assert m_segment.called
        assert not m_old_segment.called
        assert m_compose.called

    def test_process_capture_task_uses_remote_tracked_pipeline_when_enabled(self, patched_app_state):
        import capture_manager

        patched_app_state["config"]["matting_api"]["provider"] = "remote_tracked_matanyone"
        patched_app_state["config"]["tracked_matting"] = {
            "enabled": True,
            "input_frame_count": 16,
            "output_frame_indices": [3, 7, 10, 13],
            "timeout_seconds": 20,
        }
        patched_app_state["config"]["remote_matting"] = {
            "enabled": True,
            "job_timeout_seconds": 20,
        }
        patched_app_state["matting_service"] = MagicMock()
        patched_app_state["matting_service"].process_sequence.return_value = (
            [{"image_url": "/generated/final/f1.jpg", "order": 1}] * 4,
            {"remote_job_id": "job1", "upload_elapsed": 0.2, "remote_elapsed": 1.2, "download_elapsed": 0.3, "total_elapsed": 1.7},
        )

        fake_capture_data = [
            (Path("frame1.jpg"), "task_1"),
            (Path("frame2.jpg"), "task_2"),
            (Path("frame3.jpg"), "task_3"),
            (Path("frame4.jpg"), "task_4"),
        ]
        fake_sequence = {
            "capture_data": fake_capture_data,
            "capture_urls": ["u1", "u2", "u3", "u4"],
            "frame_paths": [Path("f1.jpg"), Path("f2.jpg"), Path("f3.jpg"), Path("f4.jpg")],
            "video_path": Path("v.avi"),
            "output_indices": [0, 1, 2, 3],
            "shot_task_ids": ["a", "b", "c", "d"],
        }

        with patch.object(capture_manager, "get_rotation_snapshot", return_value={"slogan": "test", "slogan_content": "test", "slogan_row": 1}), \
             patch.object(capture_manager, "get_background_items", return_value=[{"id": "bg_001", "name": "bg", "path": "x"}]), \
             patch.object(capture_manager, "select_rotating_background", return_value={"id": "bg_001", "name": "bg", "path": "x"}), \
             patch.object(capture_manager, "_capture_tracked_sequence_frames", return_value=fake_sequence) as m_capture, \
             patch.object(capture_manager, "_compose_capture_results") as m_compose:
            capture_manager.process_capture_task("task_remote")

        assert m_capture.called
        assert not m_compose.called
        assert patched_app_state["matting_service"].process_sequence.called

    def test_modelscope_universal_timeout_is_long_enough_for_cpu_matting(self, patched_app_state):
        import capture_manager

        patched_app_state["config"]["matting_api"]["provider"] = "modelscope_universal"

        assert capture_manager._resolve_capture_timeout_seconds() >= 180.0

    def test_compose_capture_results_sets_error_reason_and_instance_metrics(self, patched_app_state):
        import capture_manager

        patched_app_state["matting_service"] = MagicMock()
        patched_app_state["matting_service"].get_instance_metrics.return_value = {
            "candidates_count": 2,
            "visitors_count": 1,
            "selected_score": 0.8,
            "selected_mask_px": 12345,
            "instance_elapsed_seconds": 0.12,
        }

        with patch.object(capture_manager, "compose_single_variant", return_value={"image_url": "/generated/final/x.jpg"}):
            results = capture_manager._compose_capture_results(
                task_id="t1",
                subjects=[Image.new("RGBA", (8, 8), (0, 0, 0, 255))],
                capture_data=[(Path("f1.jpg"), "t1_1")],
                active_background={"id": "bg", "name": "bg", "path": "x"},
                slogan="test",
                slogan_row=1,
                errors_by_idx={0: "instance_segmentation_failed: no suitable person instance"},
            )

        assert results[0]["error"] is True
        assert "instance_segmentation_failed" in results[0]["error_reason"]
        assert results[0]["instance_segmentation"]["visitors_count"] == 1


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
