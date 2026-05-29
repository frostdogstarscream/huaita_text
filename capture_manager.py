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
from slogan_manager import get_rotation_snapshot, get_slogan_snapshot_by_sequence_no
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from subject_temporal_fusion import TemporalSubjectFusionConfig, fuse_subjects_temporally
from video_recorder import VideoRecorder
from remote_matting_client import RemoteMattingError
from tracked_matting_service import TrackedMattingError


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
        output_cfg = APP_STATE["config"].get("output", {})
        fallback = Image.new("RGBA", (int(output_cfg.get("width", 1080)), int(output_cfg.get("height", 1920))), (0, 0, 0, 255))
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


def _capture_video_frames(
    task_id: str,
) -> tuple[list[tuple[Path, str]], list[str]]:
    video_cfg = APP_STATE["config"].get("video_recording", {})
    if not video_cfg.get("enabled", False):
        raise RuntimeError("Video recording is not enabled in config.")

    fps = int(video_cfg.get("fps", 20))
    codec = str(video_cfg.get("codec", "mp4v"))
    post_window_s = float(video_cfg.get("post_trigger_window_s", 2.0))
    countdown_s = float(APP_STATE["config"]["laser_trigger"].get("countdown_seconds", 5))
    extract_offsets = list(video_cfg.get("extract_timestamps_s", [0.3, 0.7, 1.0, 1.5]))

    video_path = CAPTURE_DIR / f"{task_id}_video.avi"
    recorder = VideoRecorder(fps=fps, resolution=(1280, 720), codec=codec)
    recorder.start(video_path)

    total_duration = countdown_s + post_window_s
    frame_interval = 1.0 / max(fps, 1)
    update_task(task_id, status="processing", message=f"录制视频中 ({total_duration:.0f}s)...")

    start_time = time.perf_counter()
    while time.perf_counter() - start_time < total_duration:
        loop_start = time.perf_counter()
        frame = APP_STATE["camera_driver"].get_frame()
        recorder.write_frame(frame)
        elapsed = loop_start - start_time
        sleep_time = frame_interval - (time.perf_counter() - loop_start)
        if sleep_time > 0:
            time.sleep(sleep_time)

    recorder.stop()
    update_task(task_id, status="processing", message="正在从视频中抽取最佳帧...")

    capture_data: list[tuple[Path, str]] = []
    capture_urls: list[str] = []
    window_start = countdown_s  # end of countdown = start of 2s extraction window

    for idx, offset in enumerate(extract_offsets):
        shot_no = idx + 1
        frame_image = recorder.extract_frame_at(window_start + offset)
        shot_task_id = f"{task_id}_{shot_no}"
        capture_path = CAPTURE_DIR / f"{shot_task_id}.jpg"
        capture_path.parent.mkdir(parents=True, exist_ok=True)
        frame_image.save(str(capture_path), format="JPEG", quality=92)
        if not capture_path.exists() or capture_path.stat().st_size <= 0:
            raise RuntimeError(f"Failed to save extracted frame: {capture_path}")
        capture_urls.append(f"/generated/captures/{quote(f'{shot_task_id}.jpg')}")
        capture_data.append((capture_path, shot_task_id))

    return capture_data, capture_urls


def _is_tracked_matting_enabled() -> bool:
    cfg = APP_STATE["config"]
    provider = str(cfg.get("matting_api", {}).get("provider", "")).strip().lower()
    tracked_cfg = cfg.get("tracked_matting", {})
    return provider == "tracked_matanyone" and bool(tracked_cfg.get("enabled", True))


def _is_remote_tracked_matting_enabled() -> bool:
    cfg = APP_STATE["config"]
    provider = str(cfg.get("matting_api", {}).get("provider", "")).strip().lower()
    remote_cfg = cfg.get("remote_matting", {})
    return provider == "remote_tracked_matanyone" and bool(remote_cfg.get("enabled", True))


def _resolve_capture_timeout_seconds() -> float:
    cfg = APP_STATE["config"]
    provider = str(cfg.get("matting_api", {}).get("provider", "")).strip().lower()
    tracked_cfg = cfg.get("tracked_matting", {})
    if provider == "remote_tracked_matanyone":
        return max(float(cfg.get("remote_matting", {}).get("job_timeout_seconds", 20.0)), 1.0)
    if provider == "tracked_matanyone":
        return max(float(tracked_cfg.get("timeout_seconds", 20.0)), 1.0)
    if provider == "modelscope_universal":
        return max(float(cfg.get("matting_api", {}).get("timeout_seconds", 180.0)), 180.0)
    return 60.0


def _capture_tracked_sequence_frames(
    task_id: str,
    tracked_cfg: dict[str, Any],
) -> dict[str, Any]:
    frame_count = max(int(tracked_cfg.get("input_frame_count", 16)), 4)
    output_indices = tracked_cfg.get("output_frame_indices", [3, 7, 10, 13])
    output_indices = [int(i) for i in output_indices]
    if len(output_indices) != 4:
        raise RuntimeError("tracked_matting.output_frame_indices must contain 4 indices.")
    if any(i < 0 or i >= frame_count for i in output_indices):
        raise RuntimeError("tracked_matting.output_frame_indices out of range.")

    frame_interval = max(float(tracked_cfg.get("frame_interval_seconds", 0.05)), 0.0)
    video_fps = max(float(tracked_cfg.get("video_fps", 20)), 1.0)
    video_codec = str(tracked_cfg.get("video_codec", "MJPG"))

    sequence_dir = CAPTURE_DIR / f"{task_id}_tracked"
    frames_dir = sequence_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_paths: list[Path] = []
    frames: list[np.ndarray] = []
    for frame_idx in range(frame_count):
        shot_no = frame_idx + 1
        update_task(task_id, status="processing", message=f"采集中 {shot_no}/{frame_count} ...")
        frame = APP_STATE["camera_driver"].get_frame()
        frame_path = frames_dir / f"{shot_no:06d}.jpg"
        saved = cv2.imwrite(str(frame_path), frame)
        if not saved or not frame_path.exists() or frame_path.stat().st_size <= 0:
            raise RuntimeError(f"Failed to save tracked frame: {frame_path}")
        frame_paths.append(frame_path)
        frames.append(frame)
        if frame_idx < frame_count - 1 and frame_interval > 0:
            time.sleep(frame_interval)

    video_path = sequence_dir / "burst.avi"
    if not frames:
        raise RuntimeError("No frames captured for tracked matting.")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*video_codec),
        video_fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open tracked video writer: {video_path}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()

    selected_capture_data: list[tuple[Path, str]] = []
    capture_urls: list[str] = []
    shot_task_ids: list[str] = []
    for idx, frame_idx in enumerate(output_indices, start=1):
        shot_task_id = f"{task_id}_{idx}"
        shot_task_ids.append(shot_task_id)
        selected_path = frame_paths[frame_idx]
        selected_capture_data.append((selected_path, shot_task_id))
        capture_urls.append(f"/generated/captures/{quote(selected_path.name)}")

    return {
        "frame_paths": frame_paths,
        "video_path": video_path,
        "output_indices": output_indices,
        "capture_data": selected_capture_data,
        "capture_urls": capture_urls,
        "shot_task_ids": shot_task_ids,
    }


def _segment_tracked_sequence(
    task_id: str,
    sequence_capture: dict[str, Any],
) -> tuple[list[Any], list[str]]:
    matting_service = APP_STATE.get("matting_service")
    if matting_service is None:
        raise RuntimeError("Matting service is not initialized.")
    if not hasattr(matting_service, "segment_sequence"):
        raise RuntimeError("Current matting service does not support tracked sequence mode.")

    update_task(task_id, status="processing", message="跟踪抠图中 ...")
    subjects, cutout_urls, metrics = matting_service.segment_sequence(
        video_path=sequence_capture["video_path"],
        frame_paths=sequence_capture["frame_paths"],
        output_indices=sequence_capture["output_indices"],
        shot_task_ids=sequence_capture["shot_task_ids"],
        task_id=task_id,
    )
    subject_missing = float(metrics.get("subject_contact_missing_ratio", 0.0) or 0.0)
    visitor_residual = float(metrics.get("visitor_visible_residual_ratio", 0.0) or 0.0)
    elapsed_seconds = float(metrics.get("elapsed_seconds", 0.0) or 0.0)
    print(
        "[TrackedMatting] "
        f"tracking_status={metrics.get('tracking_status')} "
        f"alpha_frame_count={metrics.get('alpha_frame_count')} "
        f"subject_contact_missing_ratio={subject_missing:.4f} "
        f"visitor_visible_residual_ratio={visitor_residual:.4f} "
        f"body_outside_soft_alpha_ratio={float(metrics.get('body_outside_soft_alpha_ratio', 0.0)):.4f} "
        f"body_edge_removed_px={float(metrics.get('body_edge_removed_px', 0.0)):.1f} "
        f"body_core_missing_ratio={float(metrics.get('body_core_missing_ratio', 0.0)):.4f} "
        f"elapsed_seconds={elapsed_seconds:.2f}"
    )
    return subjects, cutout_urls


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
                output_cfg = APP_STATE["config"].get("output", {})
                subjects_by_idx[idx] = Image.new("RGBA", (int(output_cfg.get("width", 1080)), int(output_cfg.get("height", 1920))), (0, 0, 0, 255))
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
    matting_service = APP_STATE.get("matting_service")
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
        if matting_service is not None and hasattr(matting_service, "get_instance_metrics"):
            try:
                instance_metrics = matting_service.get_instance_metrics(shot_task_id)
                if instance_metrics:
                    result["instance_segmentation"] = instance_metrics
            except Exception:
                pass
        if errors_by_idx.get(i):
            result["error"] = True
            result["error_reason"] = str(errors_by_idx[i])
        results.append(result)
    return results


def _fuse_subjects_temporally_if_enabled(
    task_id: str,
    subjects: list[Any],
) -> list[Any]:
    cfg = TemporalSubjectFusionConfig.from_mapping(APP_STATE["config"].get("temporal_subject_fusion"))
    if not cfg.enabled:
        return subjects
    try:
        debug_dir = Path("generated") / "subject_debug"
        fused, report = fuse_subjects_temporally(
            subjects,
            cfg,
            debug_dir=debug_dir,
            debug_stem=task_id,
        )
        print(
            "[TemporalFusion] "
            f"alignment_success_count={report.alignment_success_count} "
            f"alpha_stable_ratio={report.alpha_stable_ratio:.4f} "
            f"removed_temporal_noise_px={report.removed_temporal_noise_px} "
            f"fallback_reason={report.fallback_reason}"
        )
        return fused
    except Exception as exc:
        print(f"[TemporalFusion] failed: {exc}")
        return subjects


def _finish_capture_task(
    task_id: str,
    snapshot: dict[str, Any],
    capture_urls: list[str],
    cutout_urls: list[str],
    results: list[dict[str, Any]],
    timed_out: bool,
    subtitle_source: str = "local_rotation",
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
        slogan_sequence_no=snapshot.get("sequence_no"),
        subtitle_source=subtitle_source,
        subtitle_playlist_id=snapshot.get("subtitle_playlist_id"),
        subtitle_revision=snapshot.get("subtitle_revision"),
        subtitle_changed_at=snapshot.get("subtitle_changed_at"),
        subtitle_received_at=snapshot.get("subtitle_received_at"),
    )


def _process_remote_tracked_results(
    task_id: str,
    sequence_capture: dict[str, Any],
    draw_slogan_text: str,
    slogan_row: int,
    active_background: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, float | str]]:
    matting_service = APP_STATE.get("matting_service")
    if matting_service is None or not hasattr(matting_service, "process_sequence"):
        raise RuntimeError("Remote matting service is not initialized.")
    update_task(task_id, status="processing", message="上传处理中...")
    results, metrics = matting_service.process_sequence(
        task_id=task_id,
        sequence_capture=sequence_capture,
        slogan=draw_slogan_text,
        slogan_row=slogan_row,
        background_item=active_background,
    )
    return results, metrics


def _resolve_capture_slogan_snapshot() -> tuple[dict[str, Any], str]:
    sync_cfg = APP_STATE["config"].get("subtitle_sync", {})
    if not sync_cfg.get("enabled", False):
        return get_rotation_snapshot(), "local_rotation"

    client = APP_STATE.get("subtitle_sync_client")
    if client is None:
        return get_rotation_snapshot(), "local_rotation"

    state = client.resolve_state_for_capture()
    snapshot = get_slogan_snapshot_by_sequence_no(state.sequence_no)
    snapshot["sequence_no"] = state.sequence_no
    snapshot["subtitle_source"] = state.source
    snapshot["subtitle_playlist_id"] = state.playlist_id
    snapshot["subtitle_revision"] = state.revision
    snapshot["subtitle_changed_at"] = state.changed_at
    snapshot["subtitle_received_at"] = state.received_at
    return snapshot, state.source


def process_capture_task(task_id: str) -> None:
    started_at = time.time()
    timeout_seconds = _resolve_capture_timeout_seconds()
    subtitle_source = "local_rotation"
    try:
        try:
            snapshot, subtitle_source = _resolve_capture_slogan_snapshot()
        except Exception as sync_exc:
            update_task(
                task_id,
                status="failed",
                message=f"字幕同步失败: {sync_exc}",
                subtitle_sync_error=str(sync_exc),
                subtitle_source="sync_failed",
                results=[],
            )
            return
        draw_slogan_text = snapshot.get("slogan_content") or snapshot["slogan"]
        slogan_row = int(snapshot.get("slogan_row", 1))
        laser_cfg = APP_STATE["config"]["laser_trigger"]
        tracked_cfg = APP_STATE["config"].get("tracked_matting", {})
        burst_count = max(int(laser_cfg.get("burst_count", 4)), 1)
        burst_interval_seconds = max(float(laser_cfg.get("burst_interval_seconds", 0.5)), 0.0)

        backgrounds = get_background_items()
        if not backgrounds:
            raise ValueError("No background templates configured.")
        active_background = select_rotating_background(backgrounds)

        if _is_remote_tracked_matting_enabled():
            sequence_capture = _capture_tracked_sequence_frames(task_id, tracked_cfg)
            capture_data = sequence_capture["capture_data"]
            capture_urls = sequence_capture["capture_urls"]
            update_task(task_id, status="processing", message="远程抠图处理中...")
            results, remote_metrics = _process_remote_tracked_results(
                task_id=task_id,
                sequence_capture=sequence_capture,
                draw_slogan_text=draw_slogan_text,
                slogan_row=slogan_row,
                active_background=active_background,
            )
            print(
                "[RemoteMatting] "
                f"remote_job_id={remote_metrics.get('remote_job_id')} "
                f"upload_elapsed={float(remote_metrics.get('upload_elapsed', 0.0)):.2f}s "
                f"remote_elapsed={float(remote_metrics.get('remote_elapsed', 0.0)):.2f}s "
                f"download_elapsed={float(remote_metrics.get('download_elapsed', 0.0)):.2f}s "
                f"total_elapsed={float(remote_metrics.get('total_elapsed', 0.0)):.2f}s"
            )
            cutout_urls: list[str] = []
            errors_by_idx: dict[int, str] = {}
            timed_out = time.time() - started_at > timeout_seconds
            _finish_capture_task(task_id, snapshot, capture_urls, cutout_urls, results, timed_out)
            return
        elif _is_tracked_matting_enabled():
            sequence_capture = _capture_tracked_sequence_frames(task_id, tracked_cfg)
            capture_data = sequence_capture["capture_data"]
            capture_urls = sequence_capture["capture_urls"]
            subjects, cutout_urls = _segment_tracked_sequence(task_id, sequence_capture)
            errors_by_idx: dict[int, str] = {}
        else:
            video_cfg = APP_STATE["config"].get("video_recording", {})
            if video_cfg.get("enabled", False):
                capture_data, capture_urls = _capture_video_frames(task_id)
            else:
                capture_data, capture_urls = _capture_burst_frames(task_id, burst_count, burst_interval_seconds)
            subjects, cutout_urls, errors_by_idx = _segment_captures_parallel(task_id, capture_data, burst_count)
            subjects = _fuse_subjects_temporally_if_enabled(task_id, subjects)

        timed_out = time.time() - started_at > timeout_seconds
        if timed_out:
            if not any(subject is not None for subject in subjects):
                update_task(task_id, status="timeout", message="当前服务繁忙，请稍后再试", results=[])
                return
            print(f"[WARN] Capture task timed out after {time.time() - started_at:.0f}s; skipping composition")

        results: list[dict[str, Any]] = []
        if not timed_out:
            update_task(task_id, status="processing", message="合成中 ...")
            results = _compose_capture_results(
                task_id,
                subjects,
                capture_data,
                active_background,
                draw_slogan_text,
                slogan_row,
                errors_by_idx,
            )

        _finish_capture_task(task_id, snapshot, capture_urls, cutout_urls, results, timed_out, subtitle_source)
    except TrackedMattingError as exc:
        update_task(task_id, status="failed", message=f"Tracked matting failed ({exc.stage}): {exc}", results=[])
    except RemoteMattingError as exc:
        update_task(task_id, status="failed", message=f"Remote matting failed ({exc.stage}): {exc}", results=[])
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
