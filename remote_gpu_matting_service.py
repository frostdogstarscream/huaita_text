from __future__ import annotations

import json
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app_state import APP_STATE, OUTPUT_DIR
from image_composer import compose_single_variant
from tracked_matting_service import TrackedMattingService


@dataclass
class RemoteJob:
    job_id: str
    status: str = "queued"
    message: str = "queued"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    results: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    task_id: str = ""
    working_dir: Path | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


JOBS: dict[str, RemoteJob] = {}
JOBS_LOCK = threading.Lock()
TRACKED_SERVICE = TrackedMattingService(APP_STATE["config"])


def _update_job(job_id: str, **values: Any) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        for key, value in values.items():
            setattr(job, key, value)
        job.updated_at = time.time()


def _safe_extract_zip(zip_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = output_dir / member.filename
            resolved = member_path.resolve()
            if not str(resolved).startswith(str(output_dir.resolve())):
                raise RuntimeError(f"Unsafe zip member path: {member.filename}")
            if member.is_dir():
                resolved.mkdir(parents=True, exist_ok=True)
                continue
            resolved.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member, "r") as src, resolved.open("wb") as dst:
                dst.write(src.read())


def _job_worker(
    *,
    job_id: str,
    zip_path: Path,
    metadata: dict[str, Any],
) -> None:
    try:
        _update_job(job_id, status="processing", message="extracting payload")
        job_dir = OUTPUT_DIR / "remote_jobs" / job_id
        _safe_extract_zip(zip_path, job_dir)
        frames_dir = job_dir / "frames"
        frame_paths = sorted(frames_dir.glob("*.jpg"))
        if len(frame_paths) < 4:
            raise RuntimeError("not enough frames in payload")
        video_path = job_dir / "burst.avi"
        if not video_path.exists():
            raise RuntimeError("burst.avi is missing in payload")

        task_id = str(metadata.get("task_id", "")).strip() or f"remote_{job_id}"
        slogan = str(metadata.get("slogan", "")).strip()
        slogan_row = int(metadata.get("slogan_row", 1))
        background_item = metadata.get("background_item", {})
        output_indices = [int(v) for v in metadata.get("output_indices", [3, 7, 10, 13])]
        shot_task_ids = [str(v) for v in metadata.get("shot_task_ids", [f"{task_id}_{i+1}" for i in range(4)])]

        if len(output_indices) != 4 or len(shot_task_ids) != 4:
            raise RuntimeError("output_indices and shot_task_ids must both contain 4 items")

        infer_started = time.perf_counter()
        subjects, _, matting_metrics = TRACKED_SERVICE.segment_sequence(
            video_path=video_path,
            frame_paths=frame_paths,
            output_indices=output_indices,
            shot_task_ids=shot_task_ids,
            task_id=task_id,
        )
        infer_elapsed_ms = (time.perf_counter() - infer_started) * 1000.0

        _update_job(job_id, message="composing finals")
        compose_started = time.perf_counter()
        results: list[dict[str, Any]] = []
        final_dir = job_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        for idx, subject in enumerate(subjects, start=1):
            result = compose_single_variant(
                subject=subject,
                slogan=slogan,
                task_id=f"{task_id}_{job_id}",
                background_item=background_item,
                order=idx,
                slogan_row=slogan_row,
            )
            filename = Path(result["image_url"]).name
            source_path = OUTPUT_DIR / "final" / filename
            target_path = final_dir / f"{idx}.jpg"
            target_path.write_bytes(source_path.read_bytes())
            results.append(
                {
                    "order": idx,
                    "image_id": result["image_id"],
                    "background_id": result["background_id"],
                    "background_name": result["background_name"],
                    "remote_image_url": f"/api/remote/jobs/{job_id}/final/{idx}",
                }
            )
        compose_elapsed_ms = (time.perf_counter() - compose_started) * 1000.0

        metrics = {
            "queue_wait_ms": 0.0,
            "infer_ms": round(infer_elapsed_ms, 2),
            "compose_ms": round(compose_elapsed_ms, 2),
            "peak_vram_mb": -1,
            "matting_metrics": matting_metrics,
        }
        _update_job(
            job_id,
            status="completed",
            message="completed",
            results=results,
            metrics=metrics,
            task_id=task_id,
            working_dir=job_dir,
        )
    except Exception as exc:
        _update_job(job_id, status="failed", message=str(exc), error=str(exc))


app = FastAPI(title="Huaita Remote Matting Service")


@app.get("/api/health")
def health() -> JSONResponse:
    return JSONResponse({"ok": True, "service": "remote_tracked_matanyone", "gpu_ready": True})


@app.post("/api/remote/jobs")
async def submit_job(payload: UploadFile = File(...), metadata: str = Form(...)) -> JSONResponse:
    job_id = uuid.uuid4().hex
    try:
        parsed_metadata = json.loads(metadata)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"invalid metadata: {exc}") from exc
    if not isinstance(parsed_metadata, dict):
        raise HTTPException(status_code=400, detail="metadata must be an object")

    job_dir = OUTPUT_DIR / "remote_jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    zip_path = job_dir / "payload.zip"
    zip_path.write_bytes(await payload.read())
    with JOBS_LOCK:
        JOBS[job_id] = RemoteJob(job_id=job_id, working_dir=job_dir, task_id=str(parsed_metadata.get("task_id", "")))

    worker = threading.Thread(
        target=_job_worker,
        kwargs={"job_id": job_id, "zip_path": zip_path, "metadata": parsed_metadata},
        daemon=True,
    )
    worker.start()
    return JSONResponse({"job_id": job_id, "status": "queued"})


@app.get("/api/remote/jobs/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        payload = {
            "job_id": job.job_id,
            "status": job.status,
            "message": job.message,
            "results": job.results,
            "error": job.error,
            "task_id": job.task_id,
            "metrics": job.metrics,
        }
    return JSONResponse(payload)


@app.get("/api/remote/jobs/{job_id}/final/{idx}")
def get_final(job_id: str, idx: int) -> FileResponse:
    if idx < 1 or idx > 4:
        raise HTTPException(status_code=400, detail="idx must be 1..4")
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        job_dir = job.working_dir
    if job_dir is None:
        raise HTTPException(status_code=404, detail="job output not found")
    image_path = job_dir / "final" / f"{idx}.jpg"
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(image_path, media_type="image/jpeg")


def run_remote_server() -> None:
    import uvicorn

    host = str(APP_STATE["config"].get("remote_matting", {}).get("host", "0.0.0.0"))
    port = int(APP_STATE["config"].get("remote_matting", {}).get("port", 18080))
    uvicorn.run("remote_gpu_matting_service:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run_remote_server()
