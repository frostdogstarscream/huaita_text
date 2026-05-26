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
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image


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


def _cutout_with_fallback(capture_path: Path, shot_task_id: str) -> tuple[Any, str | None]:
    """抠图 + 重试 + 原图兜底。始终返回 (subject, error_or_none)。"""
    import threading as _threading

    from image_composer import build_subject_cutout

    tid = _threading.current_thread().name
    t0 = time.time()
    print(f"[并发] {tid} 开始抠图 shot={shot_task_id}")

    last_error: str | None = None
    for attempt in range(3):
        try:
            subject = build_subject_cutout(capture_path, shot_task_id)
            dt = time.time() - t0
            print(f"[并发] {tid} 完成抠图 shot={shot_task_id} 耗时={dt:.1f}s 结果=成功")
            return subject, None
        except Exception as exc:
            last_error = str(exc)
            if "NotFoundFace" in last_error:
                break
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))

    try:
        fallback = Image.open(str(capture_path)).convert("RGBA")
    except Exception as exc:
        last_error = f"兜底原图也失败: {exc}"
        fallback = Image.new("RGBA", (1080, 1920), (0, 0, 0, 255))
    dt = time.time() - t0
    status = "兜底" if last_error else "成功"
    print(f"[并发] {tid} 完成抠图 shot={shot_task_id} 耗时={dt:.1f}s 结果={status}")
    return fallback, last_error


def _capture_burst_frames(
    task_id: str,
    burst_count: int,
    burst_interval_seconds: float,
) -> tuple[list[tuple[Path, str]], list[str]]:
    capture_data: list[tuple[Path, str]] = []
    capture_urls: list[str] = []
    for shot_index in range(burst_count):
        if shot_index > 0 and burst_interval_seconds > 0:
            time.sleep(burst_interval_seconds)

        shot_no = shot_index + 1
        update_task(task_id, status="processing", message=f"正在拍摄第 {shot_no}/{burst_count} 张...")
        frame = APP_STATE["camera_driver"].get_frame()

        shot_task_id = f"{task_id}_{shot_no}"
        capture_path = CAPTURE_DIR / f"{shot_task_id}.jpg"
        capture_path.parent.mkdir(parents=True, exist_ok=True)
        saved = cv2.imwrite(str(capture_path), frame)
        if not saved or not capture_path.exists() or capture_path.stat().st_size <= 0:
            raise RuntimeError(f"Failed to save capture image: {capture_path}")
        capture_urls.append(f"/generated/captures/{quote(f'{shot_task_id}.jpg')}")
        capture_data.append((capture_path, shot_task_id))
    return capture_data, capture_urls


def _segment_captures_parallel(
    task_id: str,
    capture_data: list[tuple[Path, str]],
    burst_count: int,
) -> tuple[list[Any], list[str], dict[int, str]]:
    update_task(task_id, status="processing", message=f"正在并行处理 {burst_count} 张人像分割...")
    subjects_by_idx: dict[int, Any] = {}
    errors_by_idx: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=burst_count) as executor:
        future_to_idx = {
            executor.submit(_cutout_with_fallback, cp, sid): i
            for i, (cp, sid) in enumerate(capture_data)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                subject, err = future.result()
                subjects_by_idx[idx] = subject
                if err:
                    errors_by_idx[idx] = err
                    print(f"[WARN] Cutout failed for shot {idx + 1}; using fallback: {err}")
            except Exception as exc:
                errors_by_idx[idx] = str(exc)
                subjects_by_idx[idx] = Image.new("RGBA", (1080, 1920), (0, 0, 0, 255))
                print(f"[WARN] Cutout worker failed for shot {idx + 1}: {exc}")

    if errors_by_idx:
        err_msg = "; ".join(f"shot {k + 1}: {v}" for k, v in errors_by_idx.items())
        print(f"[WARN] Some cutouts used fallback ({len(errors_by_idx)}/{burst_count}): {err_msg}")

    subjects: list[Any] = []
    cutout_urls: list[str] = []
    for i in range(len(capture_data)):
        subject = subjects_by_idx.get(i)
        if subject is not None:
            cutout_urls.append(f"/generated/cutouts/{quote(capture_data[i][1])}.png")
        subjects.append(subject)
    return subjects, cutout_urls, errors_by_idx


def _compose_capture_results(
    task_id: str,
    subjects: list[Any],
    capture_data: list[tuple[Path, str]],
    active_background: dict[str, Any],
    slogan: str,
    slogan_row: int,
    errors_by_idx: dict[int, str],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    burst_count = len(capture_data)
    for i, (subject, (_, shot_task_id)) in enumerate(zip(subjects, capture_data)):
        if subject is None:
            continue
        update_task(task_id, status="processing", message=f"正在生成第 {i + 1}/{burst_count} 张当前背景成片...")
        result = compose_single_variant(
            subject,
            slogan,
            shot_task_id,
            active_background,
            i + 1,
            slogan_row=slogan_row,
        )
        if errors_by_idx.get(i):
            result["error"] = True
        results.append(result)
    return results


def _finish_capture_task(
    task_id: str,
    snapshot: dict[str, Any],
    capture_urls: list[str],
    cutout_urls: list[str],
    results: list[dict[str, Any]],
    timed_out: bool,
) -> None:
    update_task(
        task_id,
        status="timeout" if timed_out else "completed",
        message=f"已完成 {len(results)} 张连拍成片。",
        slogan=snapshot["slogan"],
        captured_at=int(time.time()),
        capture_url=capture_urls[0] if capture_urls else None,
        capture_urls=capture_urls,
        cutout_url=cutout_urls[0] if cutout_urls else None,
        cutout_urls=cutout_urls,
        results=results,
        timed_out=timed_out,
    )


def process_capture_task(task_id: str) -> None:
    started_at = time.time()
    timeout_seconds = 60.0
    try:
        snapshot = get_rotation_snapshot()
        draw_slogan_text = snapshot.get("slogan_content") or snapshot["slogan"]
        slogan_row = int(snapshot.get("slogan_row", 1))
        laser_cfg = APP_STATE["config"]["laser_trigger"]
        burst_count = max(int(laser_cfg.get("burst_count", 4)), 1)
        burst_interval_seconds = max(float(laser_cfg.get("burst_interval_seconds", 0.5)), 0.0)

        backgrounds = get_background_items()
        if not backgrounds:
            raise ValueError("No background templates configured.")
        active_background = select_rotating_background(backgrounds)

        capture_data, capture_urls = _capture_burst_frames(task_id, burst_count, burst_interval_seconds)
        subjects, cutout_urls, errors_by_idx = _segment_captures_parallel(task_id, capture_data, burst_count)

        timed_out = time.time() - started_at > timeout_seconds
        if timed_out:
            if not any(subject is not None for subject in subjects):
                update_task(task_id, status="timeout", message="当前服务繁忙，请稍后再试", results=[])
                return
            print(f"[WARN] Capture task timed out after {time.time() - started_at:.0f}s; skipping composition")

        results: list[dict[str, Any]] = []
        if not timed_out:
            results = _compose_capture_results(
                task_id,
                subjects,
                capture_data,
                active_background,
                draw_slogan_text,
                slogan_row,
                errors_by_idx,
            )

        _finish_capture_task(task_id, snapshot, capture_urls, cutout_urls, results, timed_out)
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
