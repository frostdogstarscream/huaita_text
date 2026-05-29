"""FastAPI application: routes, lifespan, and server entry point."""

import threading
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles as _StaticFiles
from starlette.types import Scope

from ali_segment_service import AliSegmentError, AliSegmentService
from app_state import (
    APP_STATE,
    FRONTEND_DIR,
    OUTPUT_DIR,
    ensure_directories,
    ensure_kiosk_parchment_background,
    persist_laser_serial_port,
)
from background_manager import get_background_items
from camera_driver import CameraDriver, CameraFocusUnsupportedError, CameraUnavailableError, FrameUnavailableError
from capture_manager import (
    CaptureBusyError,
    build_qr_image,
    get_latest_task,
    is_camera_page_active,
    laser_trigger_loop,
    mark_camera_page_active,
    start_capture_task,
)
from laser_driver import LaserDriver
from slogan_manager import get_rotation_snapshot, get_slogan_snapshot_by_sequence_no, set_rotation_to_index
from subtitle_sync_client import SubtitleSyncClient, SubtitleSyncError


# ---------------------------------------------------------------------------
# Cache-busting: ensure browser always revalidates static assets
# ---------------------------------------------------------------------------
class _NoCacheStaticFiles(_StaticFiles):
    async def get_response(self, path: str, scope: Scope) -> Response:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


# ---------------------------------------------------------------------------
# Module-level initialization (runs once on import)
# ---------------------------------------------------------------------------
ensure_directories()
ensure_kiosk_parchment_background()
APP_STATE["camera_driver"] = CameraDriver(APP_STATE["config"]["camera"])
APP_STATE["laser_driver"] = LaserDriver(APP_STATE["config"]["laser_trigger"])
APP_STATE["laser_driver"].set_serial_port_persist_callback(persist_laser_serial_port)


def _initialize_matting_service() -> Any:
    cfg = APP_STATE["config"]
    provider = str(cfg.get("matting_api", {}).get("provider", "ali_segment_body")).strip().lower()
    if provider == "remote_tracked_matanyone":
        from remote_matting_client import RemoteTrackedMattingClient

        return RemoteTrackedMattingClient(cfg)
    if provider == "tracked_matanyone":
        from tracked_matting_service import TrackedMattingService

        return TrackedMattingService(cfg)
    return AliSegmentService(cfg)


try:
    APP_STATE["matting_service"] = _initialize_matting_service()
except (AliSegmentError, ValueError, RuntimeError) as exc:
    APP_STATE["matting_service"] = None
    print(f"[WARN] Matting service initialization failed: {exc}")


def _initialize_subtitle_sync_client() -> SubtitleSyncClient | None:
    sync_cfg = APP_STATE["config"].get("subtitle_sync", {})
    if not sync_cfg.get("enabled", False):
        return None
    try:
        return SubtitleSyncClient(sync_cfg)
    except Exception as exc:
        print(f"[WARN] Subtitle sync client initialization failed: {exc}")
        return None


APP_STATE["subtitle_sync_client"] = _initialize_subtitle_sync_client()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_: FastAPI):
    APP_STATE["camera_driver"].start()
    APP_STATE["laser_driver"].start()
    APP_STATE["laser_trigger_running"] = True
    APP_STATE["laser_trigger_worker"] = threading.Thread(target=laser_trigger_loop, daemon=True)
    APP_STATE["laser_trigger_worker"].start()

    sync_client = APP_STATE.get("subtitle_sync_client")
    if sync_client:
        APP_STATE["subtitle_sync_running"] = True
        APP_STATE["subtitle_sync_worker"] = threading.Thread(
            target=sync_client.poll_loop,
            args=(lambda: APP_STATE["subtitle_sync_running"],),
            daemon=True,
            name="subtitle-sync",
        )
        APP_STATE["subtitle_sync_worker"].start()
        print("[SubtitleSync] Background polling started.")

    try:
        yield
    finally:
        APP_STATE["subtitle_sync_running"] = False
        sync_worker = APP_STATE.get("subtitle_sync_worker")
        if sync_worker and sync_worker.is_alive():
            sync_worker.join(timeout=2)
        APP_STATE["laser_trigger_running"] = False
        worker = APP_STATE.get("laser_trigger_worker")
        if worker and worker.is_alive():
            worker.join(timeout=2)
        APP_STATE["laser_driver"].stop()
        APP_STATE["camera_driver"].stop()


# ---------------------------------------------------------------------------
# App / static mounts
# ---------------------------------------------------------------------------
app = FastAPI(title="Huaita Four Background Composer", lifespan=lifespan)
app.mount("/static", _NoCacheStaticFiles(directory=FRONTEND_DIR), name="static")
app.mount("/generated", _StaticFiles(directory=OUTPUT_DIR), name="generated")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def serve_page(name: str) -> FileResponse:
    page = FRONTEND_DIR / name
    if not page.exists():
        raise HTTPException(status_code=404, detail=f"{name} not found")
    return FileResponse(page, headers={"Cache-Control": "no-cache"})


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return serve_page("index.html")


@app.get("/index", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
def index_page() -> FileResponse:
    return serve_page("index.html")


@app.get("/select", include_in_schema=False)
@app.get("/select.html", include_in_schema=False)
def select_page() -> FileResponse:
    return serve_page("select.html")


@app.get("/view", include_in_schema=False)
@app.get("/view.html", include_in_schema=False)
def view_page() -> FileResponse:
    return serve_page("view.html")


@app.get("/download", include_in_schema=False)
@app.get("/download.html", include_in_schema=False)
def download_page() -> FileResponse:
    return serve_page("download.html")


@app.get("/camera.html", include_in_schema=False)
def camera_page() -> FileResponse:
    return serve_page("camera.html")


@app.get("/kiosk-wait.html", include_in_schema=False)
def kiosk_wait_page() -> FileResponse:
    return serve_page("kiosk-wait.html")


@app.get("/kiosk-wait", include_in_schema=False)
def kiosk_wait_redirect() -> RedirectResponse:
    return RedirectResponse(url="/kiosk-wait.html")


@app.get("/camera", include_in_schema=False)
def camera_redirect() -> RedirectResponse:
    return RedirectResponse(url="/camera.html")


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health() -> JSONResponse:
    sync_client = APP_STATE.get("subtitle_sync_client")
    subtitle_sync_status = None
    if sync_client:
        subtitle_sync_status = {
            "enabled": True,
            "connected": sync_client.status().get("connected", False),
            "sequence_no": sync_client.status().get("sequence_no"),
            "last_success_at": sync_client.status().get("last_success_at"),
            "last_error": sync_client.status().get("last_error", ""),
        }
    else:
        subtitle_sync_status = {"enabled": False}
    return JSONResponse({
        "ok": True,
        "camera": APP_STATE["camera_driver"].status(),
        "subtitle_sync": subtitle_sync_status,
    })


@app.get("/api/current-template")
def current_template() -> JSONResponse:
    sync_cfg = APP_STATE["config"].get("subtitle_sync", {})
    sync_client = APP_STATE.get("subtitle_sync_client")
    subtitle_sync_info: dict[str, Any] = {"enabled": sync_cfg.get("enabled", False)}

    if sync_cfg.get("enabled", False) and sync_client:
        subtitle_sync_info["base_url"] = sync_cfg.get("base_url", "")
        cached = sync_client.get_cached_state(max_age_seconds=10.0)
        if cached:
            try:
                snapshot = get_slogan_snapshot_by_sequence_no(cached.sequence_no)
                subtitle_sync_info.update({
                    "available": True,
                    "sequence_no": cached.sequence_no,
                    "revision": cached.revision,
                    "source": cached.source,
                })
            except ValueError:
                snapshot = get_rotation_snapshot()
                subtitle_sync_info["available"] = False
        else:
            snapshot = get_rotation_snapshot()
            subtitle_sync_info["available"] = False
    else:
        snapshot = get_rotation_snapshot()

    return JSONResponse(
        {
            "template_id": f"slogan_{snapshot['index'] + 1:03d}",
            "template_name": f"标语模板 {snapshot['index'] + 1}",
            "slogan": snapshot["slogan"],
            "slogan_row": snapshot.get("slogan_row", 1),
            "seconds_to_next": snapshot["seconds_to_next"],
            "rotation_start_time": snapshot["rotation_start_time"],
            "backgrounds": get_background_items(),
            "subtitle_sync": subtitle_sync_info,
        }
    )


@app.get("/api/ui-config")
def ui_config() -> JSONResponse:
    ui_cfg = APP_STATE["config"].get("ui", {})
    return JSONResponse(
        {
            "kiosk_idle_return_seconds": max(int(ui_cfg.get("kiosk_idle_return_seconds", 30)), 5),
            "select_background_rotate_seconds": max(int(ui_cfg.get("select_background_rotate_seconds", 10)), 3),
        }
    )


@app.get("/api/laser-status")
def laser_status() -> JSONResponse:
    payload = dict(APP_STATE["laser_driver"].status())
    payload["camera_page_active"] = is_camera_page_active()
    return JSONResponse(payload)


@app.post("/api/laser-reset")
def laser_reset() -> JSONResponse:
    return JSONResponse(APP_STATE["laser_driver"].reset_trigger_flow())


@app.post("/api/camera-page-active")
async def camera_page_active(request: Request) -> JSONResponse:
    payload = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    active = bool(payload.get("active")) if isinstance(payload, dict) else False
    mark_camera_page_active(active)
    return JSONResponse({"ok": True, "active": APP_STATE["camera_page_active"]})


@app.post("/api/sync-time")
async def sync_time(request: Request) -> JSONResponse:
    sync_cfg = APP_STATE["config"].get("subtitle_sync", {})
    if sync_cfg.get("enabled", False):
        raise HTTPException(
            status_code=409,
            detail="字幕同步已启用，有效字幕由投影服务器管理。"
        )
    payload = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    sequence_no = payload.get("sequence_no") if isinstance(payload, dict) else None
    if sequence_no in (None, ""):
        target_index = 0
    else:
        try:
            target_index = int(sequence_no) - 1
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Invalid sequence_no") from exc
    try:
        snapshot = set_rotation_to_index(target_index)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(
        {
            "ok": True,
            "rotation_start_time": snapshot["rotation_start_time"],
            "sequence_no": snapshot["index"] + 1,
            "current_template": {
                "slogan": snapshot["slogan"],
                "slogan_row": snapshot.get("slogan_row", 1),
                "seconds_to_next": snapshot["seconds_to_next"],
            },
        }
    )


@app.get("/api/subtitle-sync/status")
def subtitle_sync_status() -> JSONResponse:
    sync_client = APP_STATE.get("subtitle_sync_client")
    if sync_client is None:
        return JSONResponse({
            "enabled": False,
            "connected": False,
            "message": "字幕同步未启用",
        })
    return JSONResponse(sync_client.status())


@app.post("/api/capture")
def capture() -> JSONResponse:
    camera = APP_STATE.get("camera_driver")
    if not camera or not camera.status().get("opened") or not camera.status().get("has_frame"):
        raise HTTPException(status_code=400, detail="Camera is not ready")
    try:
        task = start_capture_task(source="manual")
    except CaptureBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JSONResponse({"task_id": task["task_id"], "status": task["status"], "message": task["message"]})


@app.get("/api/task/{task_id}")
def task_status(task_id: str) -> JSONResponse:
    with APP_STATE["tasks_lock"]:
        task = APP_STATE["tasks"].get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return JSONResponse(task)


@app.get("/api/latest-task")
def latest_task_status() -> JSONResponse:
    task = get_latest_task()
    return JSONResponse({"task": task})


@app.get("/api/camera/status")
def camera_status() -> JSONResponse:
    return JSONResponse(APP_STATE["camera_driver"].status())


@app.get("/api/camera/list")
def camera_list() -> JSONResponse:
    driver = APP_STATE["camera_driver"]
    return JSONResponse({
        "cameras": driver.list_cameras(),
        "current_index": driver.selected_index,
    })


@app.post("/api/camera/select")
async def camera_select(request: Request) -> JSONResponse:
    data = await request.json()
    idx = int(data.get("index", 0))
    return JSONResponse(APP_STATE["camera_driver"].switch_to(idx))


@app.get("/api/camera/focus")
def camera_focus_status() -> JSONResponse:
    return JSONResponse(APP_STATE["camera_driver"].focus_status())


@app.post("/api/camera/focus")
async def camera_focus_set(request: Request) -> JSONResponse:
    data = await request.json()
    auto_focus = data.get("auto_focus") if isinstance(data, dict) else None
    focus_value = data.get("focus") if isinstance(data, dict) else None
    try:
        payload = APP_STATE["camera_driver"].set_focus(
            auto_focus=None if auto_focus is None else bool(auto_focus),
            focus=None if focus_value in (None, "") else int(focus_value),
        )
    except CameraFocusUnsupportedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except CameraUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid focus value") from exc
    return JSONResponse(payload)


@app.get("/api/camera/frame")
def camera_frame() -> StreamingResponse:
    try:
        frame_bytes = APP_STATE["camera_driver"].get_frame_bytes()
    except (CameraUnavailableError, FrameUnavailableError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return StreamingResponse(iter([frame_bytes]), media_type="image/jpeg")


@app.get("/video_feed")
def video_feed() -> StreamingResponse:
    return StreamingResponse(
        APP_STATE["camera_driver"].mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/qr")
def qr_code(data: str) -> StreamingResponse:
    try:
        payload = build_qr_image(data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return StreamingResponse(BytesIO(payload), media_type="image/png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_server() -> None:
    import uvicorn

    server_cfg = APP_STATE["config"]["server"]
    uvicorn.run("main:app", host=server_cfg["host"], port=int(server_cfg["port"]), reload=False)


if __name__ == "__main__":
    run_server()

