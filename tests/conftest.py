"""Shared fixtures for all test modules."""

import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def temp_dir(tmp_path):
    """Temporary directory for file I/O tests."""
    return tmp_path


@pytest.fixture
def mock_config():
    """Minimal but complete test configuration matching DEFAULT_CONFIG structure."""
    return {
        "server": {"host": "127.0.0.1", "port": 10051},
        "camera": {
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
        },
        "rotation": {
            "interval_seconds": 30,
            "rotation_start_time": 0,
            "slogans": ["测试标语一", "测试标语二", {"content": "多行标语\n第二行", "row": 2}],
        },
        "background_set": {
            "items": [
                {
                    "id": "bg_001",
                    "name": "背景一",
                    "path": "html-page/assets/photos/1.jpg",
                    "person_layout": {"target_height_ratio": 0.62, "center_x_ratio": 0.3},
                    "text_layout": {
                        "top_overlay_height": 340,
                        "max_lines": 3,
                        "font_size_min": 56,
                        "font_size_max": 96,
                        "line_spacing_min": 6,
                        "line_spacing_max": 10,
                        "text_region": {
                            "margin_top_ratio": 0.04,
                            "width_ratio": 0.92,
                            "height_ratio": 0.21,
                        },
                    },
                },
                {
                    "id": "bg_002",
                    "name": "背景二",
                    "path": "html-page/assets/photos/2.jpg",
                },
            ]
        },
        "output": {"width": 1080, "height": 1920, "jpeg_quality": 92},
        "ui": {"kiosk_idle_return_seconds": 30, "select_background_rotate_seconds": 10},
        "person_layout": {
            "target_height_ratio": 0.72,
            "bottom_margin": 80,
            "center_x_ratio": 0.50,
            "center_y_offset": 0,
        },
        "compose": {"top_overlay_height": 340, "overlay_opacity": 120},
        "matting_api": {
            "provider": "ali_segment_body",
            "bucket": "test-bucket",
            "region": "cn-shanghai",
            "oss_endpoint": "https://oss-cn-shanghai.aliyuncs.com",
            "imageseg_endpoint": "imageseg.cn-shanghai.aliyuncs.com",
            "output_dir": "generated/cutouts",
            "max_image_edge": 2000,
        },
        "remote_matting": {
            "enabled": True,
            "base_url": "http://127.0.0.1:18080",
            "upload_mode": "video_or_zip",
            "connect_timeout_seconds": 3.0,
            "read_timeout_seconds": 20.0,
            "job_timeout_seconds": 20.0,
            "poll_interval_seconds": 0.5,
        },
        "text_style": {
            "font_size": 72,
            "top_margin": 88,
            "line_spacing": 14,
            "stroke_width": 3,
            "fill": "#FFFFFF",
            "stroke_fill": "#452d00",
            "style_mode": "gold_layered",
            "gold_palette": ["#fff7d8", "#f3cb63", "#b6751f"],
            "outline_dark": "#6c3f0a",
            "outline_width": 1,
            "shadow_color": "#4a2908",
            "shadow_offset": [1, 2],
            "shadow_blur": 1,
            "shadow_alpha": 96,
            "highlight_color": "#fffbe8",
            "highlight_offset": [0, -1],
            "highlight_alpha": 140,
            "specular_color": "#fffef4",
            "specular_strength": 188,
            "specular_band_top_ratio": 0.22,
            "specular_band_height_ratio": 0.18,
            "inner_glow_color": "#fff3b8",
            "inner_glow_alpha": 58,
        },
        "text_tuning": {
            "defaults": {
                "preferred_lines": 2,
                "max_width_ratio": 0.82,
                "auto_break_on_punctuation": True,
                "balance_weight": 1000,
            },
            "by_slogan": {},
        },
        "laser_trigger": {
            "enabled": False,
            "serial_port": "COM3",
            "baudrate": 19200,
            "bytesize": 8,
            "stopbits": 1,
            "parity": "N",
            "timeout_seconds": 0.2,
            "measure_mode": "continuous_fast_20hz",
            "trigger_min_cm": 80,
            "trigger_max_cm": 150,
            "stable_samples": 3,
            "stable_delta_cm": 5,
            "countdown_seconds": 5,
            "burst_count": 4,
            "burst_interval_seconds": 0.2,
            "cooldown_ms": 5000,
            "require_leave_before_retrigger": True,
            "leave_min_cm": 180,
        },
        "subtitle_sync": {
            "enabled": False,
            "base_url": "http://127.0.0.1:10061",
            "expected_playlist_id": "huaihai-75-v1",
            "expected_slide_count": 75,
            "poll_interval_seconds": 1.0,
            "request_timeout_seconds": 0.5,
            "max_cached_age_seconds": 1.0,
        },
    }


@pytest.fixture
def patched_app_state(mock_config):
    """Replace APP_STATE config with test config and add mock drivers.

    Also patches save_config in all modules that import it, so tests never
    write to the real config.json on disk.
    """
    import app_state
    import config_manager
    import slogan_manager

    saved_config = app_state.APP_STATE.get("config")
    saved_camera = app_state.APP_STATE.get("camera_driver")
    saved_laser = app_state.APP_STATE.get("laser_driver")
    saved_matting = app_state.APP_STATE.get("matting_service")

    app_state.APP_STATE["config"] = mock_config

    # Prevent any test from writing mock config to real config.json
    with patch.object(config_manager, "save_config", side_effect=lambda cfg: None):
        with patch.object(slogan_manager, "save_config", side_effect=lambda cfg: None):

            mock_camera = MagicMock()
            mock_camera.get_frame.return_value = None
            mock_camera.get_frame_bytes.return_value = b"fake-jpeg"
            mock_camera.status.return_value = {"running": True, "opened": True, "has_frame": True}
            mock_camera.mjpeg_generator.return_value = iter([])
            app_state.APP_STATE["camera_driver"] = mock_camera

            mock_laser = MagicMock()
            mock_laser.status.return_value = {"enabled": False, "connected": False, "trigger_state": "MANUAL_ONLY"}
            mock_laser.reset_trigger_flow.return_value = {"trigger_state": "MANUAL_ONLY"}
            mock_laser.consume_trigger.return_value = False
            app_state.APP_STATE["laser_driver"] = mock_laser

            mock_matting = MagicMock()
            app_state.APP_STATE["matting_service"] = mock_matting

            app_state.APP_STATE["tasks"] = {}
            app_state.APP_STATE["tasks_lock"] = threading.Lock()
            app_state.APP_STATE["latest_task_id"] = None
            app_state.APP_STATE["capture_busy"] = False
            app_state.APP_STATE["capture_busy_lock"] = threading.Lock()
            app_state.APP_STATE["laser_trigger_running"] = False
            app_state.APP_STATE["laser_trigger_worker"] = None
            app_state.APP_STATE["laser_trigger_error"] = ""
            app_state.APP_STATE["camera_page_active"] = False
            app_state.APP_STATE["camera_page_last_seen"] = 0.0

            yield app_state.APP_STATE

            app_state.APP_STATE["config"] = saved_config
            app_state.APP_STATE["camera_driver"] = saved_camera
            app_state.APP_STATE["laser_driver"] = saved_laser
            app_state.APP_STATE["matting_service"] = saved_matting


@pytest.fixture
def test_client():
    """FastAPI test client for the beauty inference service."""
    from fastapi.testclient import TestClient
    from beauty_service.beauty_inference_service import app, STATE
    STATE.initialize()
    return TestClient(app)
