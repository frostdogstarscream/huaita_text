"""Test API routes: HTTP status codes, response shapes, page accessibility."""

from fastapi.testclient import TestClient

import pytest


@pytest.fixture
def client(patched_app_state):
    """Create a TestClient with mocked APP_STATE."""
    from main import app
    with TestClient(app) as c:
        yield c


class TestPageRoutes:
    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_index_returns_200(self, client):
        response = client.get("/index.html")
        assert response.status_code == 200

    @pytest.mark.parametrize("path", [
        "/select.html",
        "/view.html",
        "/download.html",
        "/camera.html",
        "/kiosk-wait.html",
    ])
    def test_pages_return_200(self, client, path):
        response = client.get(path)
        assert response.status_code == 200

    def test_camera_redirect(self, client):
        response = client.get("/camera", follow_redirects=False)
        assert response.status_code in (200, 307, 302)

    def test_kiosk_wait_redirect(self, client):
        response = client.get("/kiosk-wait", follow_redirects=False)
        assert response.status_code in (200, 307, 302)

    def test_nonexistent_page_returns_404(self, client):
        response = client.get("/nonexistent.html")
        assert response.status_code == 404


class TestAPIRoutes:
    def test_health_returns_ok(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert "camera" in data

    def test_current_template(self, client):
        response = client.get("/api/current-template")
        assert response.status_code == 200
        data = response.json()
        assert "slogan" in data
        assert "backgrounds" in data
        assert "seconds_to_next" in data

    def test_ui_config(self, client):
        response = client.get("/api/ui-config")
        assert response.status_code == 200
        data = response.json()
        assert "kiosk_idle_return_seconds" in data
        assert "select_background_rotate_seconds" in data

    def test_laser_status(self, client):
        response = client.get("/api/laser-status")
        assert response.status_code == 200
        data = response.json()
        assert "enabled" in data

    def test_laser_reset(self, client):
        response = client.post("/api/laser-reset")
        assert response.status_code == 200

    def test_camera_page_active_true(self, client):
        response = client.post("/api/camera-page-active", json={"active": True})
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True

    def test_camera_page_active_false(self, client):
        response = client.post("/api/camera-page-active", json={"active": False})
        assert response.status_code == 200

    def test_sync_time(self, client):
        response = client.post("/api/sync-time", json={"sequence_no": "1"})
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert data["sequence_no"] >= 1

    def test_sync_time_invalid(self, client):
        response = client.post("/api/sync-time", json={"sequence_no": "abc"})
        assert response.status_code == 400

    def test_latest_task_returns_none_initially(self, client):
        response = client.get("/api/latest-task")
        assert response.status_code == 200
        data = response.json()
        assert data["task"] is None

    def test_camera_status(self, client):
        response = client.get("/api/camera/status")
        assert response.status_code == 200

    def test_task_not_found(self, client):
        response = client.get("/api/task/nonexistent_id")
        assert response.status_code == 404

    def test_camera_frame_available(self, client):
        response = client.get("/api/camera/frame")
        # May return 200 (mock frame) or 503 (no real camera)
        assert response.status_code in (200, 503)

    def test_video_feed(self, client):
        response = client.get("/video_feed")
        # Should return streaming response
        assert response.status_code in (200, 503)

    def test_qr_code(self, client):
        response = client.get("/api/qr?data=test")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"


class TestCaptureEndpoint:
    def test_capture_creates_task(self, client, patched_app_state):
        from unittest.mock import MagicMock
        patched_app_state["capture_busy"] = False
        patched_app_state["camera_driver"] = MagicMock()
        patched_app_state["camera_driver"].status.return_value = {"running": True, "opened": True, "has_frame": True}
        patched_app_state["camera_driver"].get_frame.return_value = None
        response = client.post("/api/capture")
        # Will be 200 (queued) or 409 (if somehow busy)
        assert response.status_code in (200, 409)
        if response.status_code == 200:
            data = response.json()
            assert "task_id" in data
            assert data["status"] in ("queued", "processing")

    def test_capture_busy_returns_409(self, client, patched_app_state):
        from unittest.mock import MagicMock
        patched_app_state["capture_busy"] = True
        patched_app_state["camera_driver"] = MagicMock()
        patched_app_state["camera_driver"].status.return_value = {"running": True, "opened": True, "has_frame": True}
        response = client.post("/api/capture")
        assert response.status_code == 409
