"""FastAPI application: routes, lifespan, and server entry point."""

import threading
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ali_segment_service import AliSegmentService
from app_state import (
    APP_STATE,
    FRONTEND_DIR,
    OUTPUT_DIR,
    ensure_directories,
    ensure_kiosk_parchment_background,
    persist_laser_serial_port,
)
from background_manager import get_background_items
from camera_driver import CameraDriver, CameraUnavailableError, FrameUnavailableError
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
from slogan_manager import get_rotation_snapshot, set_rotation_to_index


# ---------------------------------------------------------------------------
# Module-level initialization (runs once on import)
# ---------------------------------------------------------------------------
ensure_directories()
ensure_kiosk_parchment_background()
APP_STATE["camera_driver"] = CameraDriver(APP_STATE["config"]["camera"])
APP_STATE["laser_driver"] = LaserDriver(APP_STATE["config"]["laser_trigger"])
APP_STATE["laser_driver"].set_serial_port_persist_callback(persist_laser_serial_port)
APP_STATE["matting_service"] = AliSegmentService(APP_STATE["config"])


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
    try:
        yield
    finally:
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
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
app.mount("/generated", StaticFiles(directory=OUTPUT_DIR), name="generated")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def serve_page(name: str) -> FileResponse:
    page = FRONTEND_DIR / name
    if not page.exists():
        raise HTTPException(status_code=404, detail=f"{name} not found")
    return FileResponse(page)


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
    return JSONResponse({"ok": True, "camera": APP_STATE["camera_driver"].status()})


@app.get("/api/current-template")
def current_template() -> JSONResponse:
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


@app.post("/api/capture")
def capture() -> JSONResponse:
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
