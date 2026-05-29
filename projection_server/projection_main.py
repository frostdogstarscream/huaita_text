from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from starlette.responses import JSONResponse

from projection_server.projection_app_state import (
    PROJECTION_STATE,
    advance_slide,
    get_subtitle_state,
    set_slide,
    toggle_playing,
)


def _rotation_loop() -> None:
    while PROJECTION_STATE["running"]:
        interval = PROJECTION_STATE["config"].get("interval_seconds", 5)
        time.sleep(interval)
        if not PROJECTION_STATE["running"]:
            break
        if PROJECTION_STATE["playing"]:
            advance_slide()


@asynccontextmanager
async def lifespan(app: FastAPI):
    PROJECTION_STATE["running"] = True
    worker = threading.Thread(target=_rotation_loop, daemon=True, name="projection-rotation")
    PROJECTION_STATE["rotation_worker"] = worker
    worker.start()
    yield
    PROJECTION_STATE["running"] = False
    worker.join(timeout=3)


app = FastAPI(title="Projection Server", lifespan=lifespan)


@app.get("/api/health")
async def health():
    return JSONResponse({
        "ok": True,
        "playlist_valid": PROJECTION_STATE["playlist_valid"],
    })


@app.get("/api/subtitle-state")
async def subtitle_state():
    return JSONResponse(get_subtitle_state())


@app.post("/api/control/play")
async def control_play():
    playing = toggle_playing()
    return JSONResponse({"playing": playing})


@app.post("/api/control/next")
async def control_next():
    advance_slide()
    return JSONResponse(get_subtitle_state())


@app.post("/api/control/prev")
async def control_prev():
    with PROJECTION_STATE["state_lock"]:
        idx = PROJECTION_STATE["current_index"] - 1
    set_slide(idx if idx >= 0 else PROJECTION_STATE["slide_count"] - 1)
    return JSONResponse(get_subtitle_state())


@app.post("/api/control/goto/{sequence_no}")
async def control_goto(sequence_no: int):
    set_slide(sequence_no - 1)
    return JSONResponse(get_subtitle_state())


@app.get("/api/current-slide-image")
async def current_slide_image():
    with PROJECTION_STATE["state_lock"]:
        paths = PROJECTION_STATE["slide_paths"]
        idx = PROJECTION_STATE["current_index"]
    if not paths or idx < 0 or idx >= len(paths):
        return JSONResponse({"error": "no slide available"}, status_code=404)
    return FileResponse(paths[idx], headers={"Cache-Control": "no-cache"})


def run_server(host: str = "0.0.0.0", port: int = 10061) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_config=None)


if __name__ == "__main__":
    cfg = PROJECTION_STATE["config"]["server"]
    run_server(cfg["host"], cfg["port"])
