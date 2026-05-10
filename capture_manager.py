"""Capture workflow: task management, photo capture, laser trigger loop, QR generation."""

import threading
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import quote

import cv2
import numpy as np

from app_state import APP_STATE, CAMERA_PAGE_HEARTBEAT_TIMEOUT_SECONDS, CAPTURE_DIR
from background_manager import get_background_items, select_rotating_background
from config_manager import normalize_text_tree
from image_composer import build_subject_cutout, compose_single_variant
from slogan_manager import get_rotation_snapshot


class CaptureBusyError(RuntimeError):
    pass


def update_task(task_key: str, **values: Any) -> None:
    with APP_STATE["tasks_lock"]:
        task = APP_STATE["tasks"].setdefault(task_key, {})
        task.update(normalize_text_tree(values))


def get_latest_task() -> dict[str, Any] | None:
    with APP_STATE["tasks_lock"]:
        latest_task_id = APP_STATE.get("latest_task_id")
        if not latest_task_id:
            return None
        task = APP_STATE["tasks"].get(latest_task_id)
        if not task:
            return None
        return dict(task)


def is_capture_busy() -> bool:
    with APP_STATE["capture_busy_lock"]:
        return bool(APP_STATE["capture_busy"])


def try_acquire_capture_slot() -> bool:
    with APP_STATE["capture_busy_lock"]:
        if APP_STATE["capture_busy"]:
            return False
        APP_STATE["capture_busy"] = True
        return True


def release_capture_slot() -> None:
    with APP_STATE["capture_busy_lock"]:
        APP_STATE["capture_busy"] = False


def process_capture_task(task_id: str) -> None:
    try:
        with APP_STATE["tasks_lock"]:
            trigger_source = APP_STATE["tasks"].get(task_id, {}).get("trigger_source", "manual")
        snapshot = get_rotation_snapshot()
        draw_slogan_text = snapshot.get("slogan_content") or snapshot["slogan"]
        slogan_row = int(snapshot.get("slogan_row", 1))
        laser_cfg = APP_STATE["config"]["laser_trigger"]
        burst_count = max(int(laser_cfg.get("burst_count", 4)), 1)
        if trigger_source != "laser":
            frame = APP_STATE["camera_driver"].get_frame()

            capture_name = f"{task_id}.jpg"
            capture_path = CAPTURE_DIR / capture_name
            cv2.imwrite(str(capture_path), frame)

            update_task(task_id, status="processing", message="正在调用阿里云人像分割…")
            subject = build_subject_cutout(capture_path, task_id)
            update_task(task_id, status="processing", message="正在解析前景透明图…")
            backgrounds = get_background_items()
            active_background = select_rotating_background(backgrounds)
            update_task(task_id, status="processing", message="正在生成当前轮换背景成片…")
            results: list[dict[str, Any]] = []
            for order in range(1, burst_count + 1):
                results.append(
                    compose_single_variant(
                        subject,
                        draw_slogan_text,
                        task_id,
                        active_background,
                        order,
                        slogan_row=slogan_row,
                    )
                )
            update_task(
                task_id,
                status="completed",
                message=f"已生成 {len(results)} 张当前背景融合图。",
                slogan=snapshot["slogan"],
                captured_at=int(time.time()),
                capture_url=f"/generated/captures/{quote(capture_name)}",
                cutout_url=f"/generated/cutouts/{quote(task_id)}.png",
                results=results,
            )
            return

        burst_interval_seconds = max(float(laser_cfg.get("burst_interval_seconds", 0.2)), 0.0)
        backgrounds = get_background_items()
        if not backgrounds:
            raise ValueError("No background templates configured.")
        active_background = select_rotating_background(backgrounds)

        results: list[dict[str, Any]] = []
        capture_urls: list[str] = []
        cutout_urls: list[str] = []

        order = 1
        for shot_index in range(burst_count):
            if shot_index > 0 and burst_interval_seconds > 0:
                time.sleep(burst_interval_seconds)

            shot_no = shot_index + 1
            update_task(task_id, status="processing", message=f"正在拍摄第 {shot_no}/{burst_count} 张…")
            frame = APP_STATE["camera_driver"].get_frame()

            shot_task_id = f"{task_id}_{shot_no}"
            capture_name = f"{shot_task_id}.jpg"
            capture_path = CAPTURE_DIR / capture_name
            cv2.imwrite(str(capture_path), frame)
            capture_urls.append(f"/generated/captures/{quote(capture_name)}")

            update_task(task_id, status="processing", message=f"正在处理第 {shot_no}/{burst_count} 张人像分割…")
            subject = build_subject_cutout(capture_path, shot_task_id)
            cutout_urls.append(f"/generated/cutouts/{quote(shot_task_id)}.png")

            update_task(task_id, status="processing", message=f"正在生成第 {shot_no}/{burst_count} 张当前背景成片…")
            results.append(
                compose_single_variant(
                    subject,
                    draw_slogan_text,
                    shot_task_id,
                    active_background,
                    order,
                    slogan_row=slogan_row,
                )
            )
            order += 1

        update_task(
            task_id,
            status="completed",
            message=f"已完成 {len(results)} 张连拍成片。",
            slogan=snapshot["slogan"],
            captured_at=int(time.time()),
            capture_url=capture_urls[0] if capture_urls else None,
            capture_urls=capture_urls,
            cutout_url=cutout_urls[0] if cutout_urls else None,
            cutout_urls=cutout_urls,
            results=results,
        )
    except Exception as exc:
        update_task(task_id, status="failed", message=str(exc), results=[])
    finally:
        release_capture_slot()


def legacy_start_capture_task() -> dict[str, Any]:
    task_id = uuid.uuid4().hex
    update_task(task_id, task_id=task_id, status="queued", message="任务已创建，等待处理。", results=[])
    APP_STATE["latest_task_id"] = task_id
    worker = threading.Thread(target=process_capture_task, args=(task_id,), daemon=True)
    worker.start()
    return APP_STATE["tasks"][task_id]


def start_capture_task(source: str = "manual") -> dict[str, Any]:
    if not try_acquire_capture_slot():
        raise CaptureBusyError("Capture task is already in progress.")

    task_id = uuid.uuid4().hex
    try:
        update_task(
            task_id,
            task_id=task_id,
            status="queued",
            message="任务已创建，等待处理。",
            results=[],
            trigger_source=source,
        )
        APP_STATE["latest_task_id"] = task_id
        worker = threading.Thread(target=process_capture_task, args=(task_id,), daemon=True)
        worker.start()
        return APP_STATE["tasks"][task_id]
    except Exception:
        release_capture_slot()
        raise


def mark_camera_page_active(active: bool) -> None:
    APP_STATE["camera_page_active"] = bool(active)
    APP_STATE["camera_page_last_seen"] = time.time() if active else 0.0


def is_camera_page_active() -> bool:
    if not APP_STATE.get("camera_page_active"):
        return False
    last_seen = float(APP_STATE.get("camera_page_last_seen") or 0.0)
    if time.time() - last_seen > CAMERA_PAGE_HEARTBEAT_TIMEOUT_SECONDS:
        APP_STATE["camera_page_active"] = False
        return False
    return True


def laser_trigger_loop() -> None:
    while APP_STATE["laser_trigger_running"]:
        try:
            APP_STATE["laser_driver"].tick()
            if APP_STATE["laser_driver"].consume_trigger() and is_camera_page_active() and not is_capture_busy():
                start_capture_task(source="laser")
        except CaptureBusyError:
            pass
        except Exception as exc:
            APP_STATE["laser_trigger_error"] = str(exc)
        time.sleep(0.1)


def build_qr_image(payload: str) -> bytes:
    params = cv2.QRCodeEncoder_Params()
    encoder = cv2.QRCodeEncoder_create(params)
    matrix = encoder.encode(payload)
    if matrix is None:
        raise ValueError("Failed to generate QR code.")
    qr = np.where(matrix == 0, 0, 255).astype("uint8")
    qr = cv2.resize(qr, (280, 280), interpolation=cv2.INTER_NEAREST)
    qr_rgb = cv2.cvtColor(qr, cv2.COLOR_GRAY2RGB)
    ok, encoded = cv2.imencode(".png", qr_rgb)
    if not ok:
        raise ValueError("Failed to encode QR image.")
    return encoded.tobytes()
