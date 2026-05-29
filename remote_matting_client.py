from __future__ import annotations

import json
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from app_state import FINAL_DIR


class RemoteMattingError(RuntimeError):
    def __init__(self, message: str, *, stage: str = "remote_matting") -> None:
        super().__init__(message)
        self.stage = stage


@dataclass(frozen=True)
class RemoteMattingConfig:
    enabled: bool = True
    base_url: str = "http://127.0.0.1:18080"
    upload_mode: str = "video_or_zip"
    connect_timeout_seconds: float = 3.0
    read_timeout_seconds: float = 20.0
    job_timeout_seconds: float = 20.0
    poll_interval_seconds: float = 0.5

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> "RemoteMattingConfig":
        data = mapping or {}
        return cls(
            enabled=bool(data.get("enabled", True)),
            base_url=str(data.get("base_url", "http://127.0.0.1:18080")).rstrip("/"),
            upload_mode=str(data.get("upload_mode", "video_or_zip")),
            connect_timeout_seconds=max(float(data.get("connect_timeout_seconds", 3.0)), 0.1),
            read_timeout_seconds=max(float(data.get("read_timeout_seconds", 20.0)), 0.1),
            job_timeout_seconds=max(float(data.get("job_timeout_seconds", 20.0)), 1.0),
            poll_interval_seconds=max(float(data.get("poll_interval_seconds", 0.5)), 0.1),
        )


class RemoteTrackedMattingClient:
    def __init__(self, config: dict[str, Any]) -> None:
        self.provider = "remote_tracked_matanyone"
        self.cfg = RemoteMattingConfig.from_mapping(config.get("remote_matting", {}))

    def _make_zip(self, sequence_capture: dict[str, Any], task_id: str) -> Path:
        sequence_dir = Path(sequence_capture["video_path"]).parent
        zip_path = sequence_dir / f"{task_id}_remote_payload.zip"
        frame_paths: list[Path] = [Path(p) for p in sequence_capture["frame_paths"]]
        video_path = Path(sequence_capture["video_path"])
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            if video_path.exists():
                zf.write(video_path, arcname="burst.avi")
            for frame_path in frame_paths:
                zf.write(frame_path, arcname=f"frames/{frame_path.name}")
        return zip_path

    def process_sequence(
        self,
        *,
        task_id: str,
        sequence_capture: dict[str, Any],
        slogan: str,
        slogan_row: int,
        background_item: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, float | str]]:
        started = time.perf_counter()
        zip_path = self._make_zip(sequence_capture, task_id)
        metadata = {
            "task_id": task_id,
            "slogan": slogan,
            "slogan_row": slogan_row,
            "background_item": background_item,
            "output_indices": sequence_capture["output_indices"],
            "shot_task_ids": sequence_capture["shot_task_ids"],
        }
        upload_started = time.perf_counter()
        try:
            with zip_path.open("rb") as payload:
                response = requests.post(
                    f"{self.cfg.base_url}/api/remote/jobs",
                    files={"payload": (zip_path.name, payload, "application/zip")},
                    data={"metadata": json.dumps(metadata, ensure_ascii=False)},
                    timeout=(self.cfg.connect_timeout_seconds, self.cfg.read_timeout_seconds),
                )
        except requests.RequestException as exc:
            raise RemoteMattingError(f"remote upload failed: {exc}", stage="upload") from exc
        if response.status_code >= 400:
            raise RemoteMattingError(f"remote upload failed: HTTP {response.status_code}", stage="upload")
        body = response.json()
        remote_job_id = str(body.get("job_id", "")).strip()
        if not remote_job_id:
            raise RemoteMattingError("remote upload failed: missing job_id", stage="upload")
        upload_elapsed = time.perf_counter() - upload_started

        poll_started = time.perf_counter()
        status_payload: dict[str, Any] = {}
        while True:
            elapsed = time.perf_counter() - poll_started
            if elapsed > self.cfg.job_timeout_seconds:
                raise RemoteMattingError(
                    f"remote job timed out after {self.cfg.job_timeout_seconds:.1f}s",
                    stage="timeout",
                )
            try:
                poll_resp = requests.get(
                    f"{self.cfg.base_url}/api/remote/jobs/{quote(remote_job_id)}",
                    timeout=(self.cfg.connect_timeout_seconds, self.cfg.read_timeout_seconds),
                )
            except requests.RequestException as exc:
                raise RemoteMattingError(f"remote poll failed: {exc}", stage="poll") from exc
            if poll_resp.status_code >= 400:
                raise RemoteMattingError(f"remote poll failed: HTTP {poll_resp.status_code}", stage="poll")
            status_payload = poll_resp.json()
            status = str(status_payload.get("status", ""))
            if status == "completed":
                break
            if status in {"failed", "timeout"}:
                raise RemoteMattingError(str(status_payload.get("message", "remote job failed")), stage=status)
            time.sleep(self.cfg.poll_interval_seconds)
        remote_elapsed = time.perf_counter() - poll_started

        download_started = time.perf_counter()
        remote_results = status_payload.get("results", [])
        if not isinstance(remote_results, list) or len(remote_results) != 4:
            raise RemoteMattingError("remote completed but results are invalid", stage="download")
        local_results: list[dict[str, Any]] = []
        for idx, item in enumerate(remote_results, start=1):
            image_id = str(item.get("image_id") or f"remote_{idx}")
            remote_url = str(item.get("remote_image_url", "")).strip()
            if not remote_url:
                raise RemoteMattingError("remote result missing remote_image_url", stage="download")
            try:
                image_resp = requests.get(
                    f"{self.cfg.base_url}{remote_url}",
                    timeout=(self.cfg.connect_timeout_seconds, self.cfg.read_timeout_seconds),
                )
            except requests.RequestException as exc:
                raise RemoteMattingError(f"remote download failed: {exc}", stage="download") from exc
            if image_resp.status_code >= 400:
                raise RemoteMattingError(
                    f"remote download failed: HTTP {image_resp.status_code}",
                    stage="download",
                )
            filename = f"{task_id}_{idx}_{image_id}.jpg"
            output_path = FINAL_DIR / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(image_resp.content)
            local_results.append(
                {
                    "image_id": image_id,
                    "image_url": f"/generated/final/{quote(filename)}",
                    "background_id": item.get("background_id"),
                    "background_name": item.get("background_name"),
                    "order": idx,
                }
            )
        download_elapsed = time.perf_counter() - download_started
        total_elapsed = time.perf_counter() - started
        metrics: dict[str, float | str] = {
            "remote_job_id": remote_job_id,
            "upload_elapsed": upload_elapsed,
            "remote_elapsed": remote_elapsed,
            "download_elapsed": download_elapsed,
            "total_elapsed": total_elapsed,
        }
        return local_results, metrics
