import asyncio
import traceback
import threading
import time
from datetime import datetime
from typing import Any, Generator

import cv2
import numpy as np

from runtime_paths import get_app_paths


class CameraUnavailableError(RuntimeError):
    pass


class FrameUnavailableError(RuntimeError):
    pass


class CameraFocusUnsupportedError(RuntimeError):
    pass


BACKEND_MAP = {
    "CAP_ANY": cv2.CAP_ANY,
    "CAP_DSHOW": cv2.CAP_DSHOW,
    "CAP_MSMF": cv2.CAP_MSMF,
}


class CameraDriver:
    def __init__(self, camera_config: dict[str, Any]) -> None:
        self.index = int(camera_config["index"])
        self.backend_name = str(camera_config["backend"]).upper()
        self.backend = BACKEND_MAP.get(self.backend_name, cv2.CAP_ANY)
        self.selection_mode = str(camera_config.get("selection_mode", "fixed")).lower()
        self.probe_indices = [int(value) for value in camera_config.get("probe_indices", [0, 1, 2, 3, 4, 5])]
        self.preferred_indices = [int(value) for value in camera_config.get("preferred_indices", [2, 1, 3, 4, 5, 0])]
        configured_backend_order = camera_config.get("backend_order")
        if configured_backend_order:
            self.backend_order = [str(value).upper() for value in configured_backend_order]
        else:
            self.backend_order = [self.backend_name, "CAP_ANY", "CAP_MSMF"]
        self.backend_order = self._dedupe_backends(self.backend_order)
        self.width = int(camera_config["width"])
        self.height = int(camera_config["height"])
        self.fps = int(camera_config["fps"])
        self.auto_focus = bool(camera_config.get("auto_focus", True))
        manual_focus = camera_config.get("manual_focus")
        self.manual_focus = None if manual_focus in (None, "") else int(manual_focus)
        self.focus_probe_values = [
            int(value) for value in camera_config.get("focus_probe_values", [0, 50, 100, 150, 200, 250])
        ]
        self.jpeg_quality = int(camera_config["jpeg_quality"])
        self.log_enabled = bool(camera_config.get("log_enabled", True))
        configured_log_path = str(camera_config.get("log_path", "") or "").strip()
        self.log_path = configured_log_path or str(get_app_paths()["base_dir"] / "huaita_camera.log")
        self.stale_frame_seconds = float(camera_config.get("stale_frame_seconds", 5.0))
        self.read_fail_count = 0
        self.restart_count = 0
        self._last_stale_log_time = 0.0
        self.auto_focus_set_ok: bool | None = None
        self.auto_focus_last_error = ""
        self.auto_focus_supported = False
        self.focus_supported = False
        self.focus_value: float | None = None
        self.focus_reported_value: float | None = None
        self.focus_min: int | None = None
        self.focus_max: int | None = None
        self.focus_step: int | None = None
        self.focus_set_ok: bool | None = None
        self.focus_last_error = ""

        self.capture: cv2.VideoCapture | None = None
        self.frame: np.ndarray | None = None
        self.frame_lock = threading.Lock()
        self.control_lock = threading.RLock()
        self.running = False
        self.last_error = "camera not started"
        self.last_frame_time = 0.0
        self.selected_index: int | None = None
        self.selected_backend: str | None = None
        self.probe_attempts = 0
        self.probe_results: list[dict[str, Any]] = []
        self._camera_list_cache: list[dict[str, Any]] | None = None
        self._camera_list_cache_time = 0.0
        self._needs_restart = False
        self._reader_handle: asyncio.TimerHandle | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._startup_thread: threading.Thread | None = None
        self._log(
            "camera driver initialized "
            f"selection_mode={self.selection_mode} index={self.index} backend={self.backend_name} "
            f"probe_indices={self.probe_indices} preferred_indices={self.preferred_indices} "
            f"backend_order={self.backend_order} size={self.width}x{self.height} fps={self.fps} "
            f"auto_focus={self.auto_focus} manual_focus={self.manual_focus}"
        )

    def start(self) -> None:
        if self.running:
            self._log("start requested but camera driver is already running")
            return
        self.running = True
        self._loop = asyncio.get_running_loop()
        self._log("camera driver start requested")
        self._startup_thread = threading.Thread(target=self._init_and_start_reading, name="huaita-camera-start", daemon=True)
        self._startup_thread.start()

    def stop(self) -> None:
        self._log("camera driver stop requested")
        self.running = False
        if self._reader_handle is not None:
            self._reader_handle.cancel()
            self._reader_handle = None
        self._release_capture()

    def try_restart(self) -> None:
        """从主线程重新探测摄像头（供 FastAPI 后台任务或 asyncio 回调调用）。"""
        if self.running and (not self.capture or not self.capture.isOpened()):
            self.restart_count += 1
            self._log(f"camera restart attempt #{self.restart_count}")
            self._open_capture()
            if self.capture and self.capture.isOpened():
                self._needs_restart = False
                self._log(f"camera restart succeeded index={self.selected_index} backend={self.selected_backend}")
            else:
                self._log(f"camera restart failed last_error={self.last_error}")

    # ------------------------------------------------------------------
    # 帧读取回调（运行在 asyncio event loop 线程 = 主线程）
    # ------------------------------------------------------------------

    def _init_and_start_reading(self) -> None:
        """首次探测摄像头然后启动帧读取循环。"""
        if not self.running:
            self._log("startup thread exited because driver is not running")
            return
        self._log("startup thread opening camera")
        self._open_capture()
        if self.running and self._loop is not None:
            self._loop.call_soon_threadsafe(self._schedule_next_read)
            self._log("camera read loop scheduled")

    def _schedule_next_read(self) -> None:
        """调度下一次帧读取（约 30 fps）。"""
        if self.running and self._loop is not None:
            self._reader_handle = self._loop.call_later(0.033, self._read_frame)

    def _read_frame(self) -> None:
        """在主线程读取一帧摄像头画面。"""
        if not self.running:
            return
        try:
            if not self.capture or not self.capture.isOpened():
                self.try_restart()
                self._schedule_next_read()
                return

            with self.control_lock:
                if not self.capture or not self.capture.isOpened():
                    self._schedule_next_read()
                    return
                for _ in range(3):
                    self.capture.grab()
                ok, frame = self.capture.retrieve()
            if ok and frame is not None:
                had_error = bool(self.last_error)
                with self.frame_lock:
                    self.frame = frame
                    self.last_frame_time = time.time()
                    self.last_error = ""
                if had_error:
                    self._log(f"camera frame recovered shape={tuple(frame.shape)}")
            else:
                self.read_fail_count += 1
                self.last_error = "read failed"
                self._log(f"camera read failed count={self.read_fail_count}; releasing capture")
                self._release_capture()
        except Exception as exc:
            self.read_fail_count += 1
            self.last_error = f"read frame error: {exc}"
            self._log(f"camera read exception count={self.read_fail_count}: {exc}", exc=exc)
            self._release_capture()
        finally:
            self._restart_if_frame_stale()
            self._schedule_next_read()

    def status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "index": self.index,
            "backend": self.backend_name,
            "selection_mode": self.selection_mode,
            "selected_index": self.selected_index,
            "selected_backend": self.selected_backend,
            "probe_indices": list(self.probe_indices),
            "preferred_indices": list(self.preferred_indices),
            "backend_order": list(self.backend_order),
            "probe_attempts": self.probe_attempts,
            "probe_results": list(self.probe_results),
            "opened": bool(self.capture and self.capture.isOpened()),
            "last_frame_time": self.last_frame_time,
            "seconds_since_last_frame": self._seconds_since_last_frame(),
            "last_error": self.last_error,
            "has_frame": self.frame is not None,
            "log_enabled": self.log_enabled,
            "log_path": self.log_path,
            "stale_frame_seconds": self.stale_frame_seconds,
            "read_fail_count": self.read_fail_count,
            "restart_count": self.restart_count,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "auto_focus": self.auto_focus,
            "auto_focus_set_ok": self.auto_focus_set_ok,
            "auto_focus_last_error": self.auto_focus_last_error,
            "auto_focus_supported": self.auto_focus_supported,
            "focus_supported": self.focus_supported,
            "focus_value": self.focus_value,
            "focus_reported_value": self.focus_reported_value,
            "manual_focus": self.manual_focus,
            "focus_min": self.focus_min,
            "focus_max": self.focus_max,
            "focus_step": self.focus_step,
            "focus_set_ok": self.focus_set_ok,
            "focus_last_error": self.focus_last_error,
        }

    def focus_status(self) -> dict[str, Any]:
        with self.control_lock:
            if self.capture and self.capture.isOpened():
                self._refresh_focus_state("")
            return {
                "opened": bool(self.capture and self.capture.isOpened()),
                "selected_index": self.selected_index,
                "selected_backend": self.selected_backend,
                "auto_focus": self.auto_focus,
                "auto_focus_supported": self.auto_focus_supported,
                "auto_focus_set_ok": self.auto_focus_set_ok,
                "auto_focus_last_error": self.auto_focus_last_error,
                "focus_supported": self.focus_supported,
                "focus": self.focus_value,
                "focus_reported": self.focus_reported_value,
                "manual_focus": self.manual_focus,
                "focus_min": self.focus_min,
                "focus_max": self.focus_max,
                "focus_step": self.focus_step,
                "focus_set_ok": self.focus_set_ok,
                "focus_last_error": self.focus_last_error,
            }

    def set_focus(self, *, auto_focus: bool | None = None, focus: int | None = None) -> dict[str, Any]:
        with self.control_lock:
            if not self.capture or not self.capture.isOpened():
                raise CameraUnavailableError("Camera is not opened.")

            errors: list[str] = []
            if auto_focus is not None:
                try:
                    ok = bool(self.capture.set(cv2.CAP_PROP_AUTOFOCUS, 1 if auto_focus else 0))
                    self.auto_focus = bool(auto_focus)
                    self.auto_focus_set_ok = ok
                    if not ok:
                        errors.append("auto focus control was rejected by camera driver")
                except Exception as exc:
                    self.auto_focus_set_ok = False
                    errors.append(f"auto focus error: {exc}")

            if focus is not None:
                self._refresh_focus_state("")
                if not self.focus_supported:
                    self.focus_last_error = "manual focus is not exposed by this camera driver"
                    raise CameraFocusUnsupportedError(self.focus_last_error)
                try:
                    self.capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                except Exception:
                    pass
                try:
                    ok = bool(self.capture.set(cv2.CAP_PROP_FOCUS, int(focus)))
                    self.manual_focus = int(focus)
                    self.auto_focus = False
                    self.focus_set_ok = ok
                    if not ok:
                        errors.append("manual focus control was rejected by camera driver")
                except Exception as exc:
                    self.focus_set_ok = False
                    errors.append(f"manual focus error: {exc}")

            self._refresh_focus_state("; ".join(errors))
            return self.focus_status()

    def list_cameras(self) -> list[dict[str, Any]]:
        """探测可用摄像头（5 秒缓存，避免频繁探测冲垮后端）。"""
        now = time.time()
        if self._camera_list_cache and (now - self._camera_list_cache_time) < 5.0:
            # 刷新 current 标记（可能因 reader loop 切换而改变）
            current = self.selected_index
            for item in self._camera_list_cache:
                item["current"] = item["index"] == current
            return self._camera_list_cache

        results: list[dict[str, Any]] = []
        current = self.selected_index
        for idx in range(6):
            # 当前已选中的摄像头跳过重探测（正被 reader loop 占用）
            if idx == current and self.capture and self.capture.isOpened():
                results.append({
                    "index": idx,
                    "working": True,
                    "name": f"Camera {idx} ({self.selected_backend or 'active'})",
                    "current": True,
                })
                continue
            working = False
            name = f"Camera {idx}"
            # 仅用 CAP_ANY（系统默认后端），避免遍历所有后端造成冲突
            try:
                cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
            except Exception:
                cap = None
            if cap and cap.isOpened():
                try:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    ok, _frame = cap.read()
                except Exception:
                    ok = False
                if ok:
                    working = True
                    try:
                        be = cap.getBackendName() if hasattr(cap, 'getBackendName') else ""
                        name = f"Camera {idx} ({be})".strip()
                    except Exception:
                        name = f"Camera {idx}"
                cap.release()
            results.append({
                "index": idx,
                "working": working,
                "name": name,
                "current": idx == current,
            })
        self._camera_list_cache = results
        self._camera_list_cache_time = now
        return results

    def switch_to(self, index: int) -> dict[str, Any]:
        """切换到指定摄像头索引，重新打开 capture。"""
        self.index = int(index)
        self.selection_mode = "fixed"
        self._release_capture()
        self._open_capture()
        return self.status()

    def get_frame(self) -> np.ndarray:
        if not self.running:
            raise CameraUnavailableError("Camera driver is not running.")
        with self.frame_lock:
            if self.frame is None:
                raise FrameUnavailableError("Camera frame is not available yet.")
            return self.frame.copy()

    def get_frame_bytes(self) -> bytes:
        frame = self.get_frame()
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            raise FrameUnavailableError("Failed to encode camera frame.")
        return encoded.tobytes()

    def mjpeg_generator(self) -> Generator[bytes, None, None]:
        while True:
            try:
                frame_bytes = self.get_frame_bytes()
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                time.sleep(max(1 / max(self.fps, 1), 0.04))
            except (CameraUnavailableError, FrameUnavailableError):
                time.sleep(0.3)

    def _open_capture(self) -> None:
        try:
            with self.control_lock:
                self._release_capture()
                self.probe_attempts += 1
                self.selected_index = None
                self.selected_backend = None
            candidates = self._build_candidate_indices()
            self._log(
                f"open capture attempt #{self.probe_attempts} candidates={candidates} "
                f"backend_order={self.backend_order}"
            )
            probe_results: list[dict[str, Any]] = []

            for candidate_index in candidates:
                for backend_name in self.backend_order:
                    self._log(f"probing camera index={candidate_index} backend={backend_name}")
                    try:
                        result = self._probe_camera(candidate_index, backend_name)
                    except Exception as exc:
                        self._log(
                            f"camera probe exception index={candidate_index} backend={backend_name}: {exc}",
                            exc=exc,
                        )
                        result = {
                            "index": candidate_index, "backend": backend_name,
                            "opened": False, "read_ok": False, "shape": None,
                            "error": f"C++ exception: {exc}", "capture": None, "frame": None,
                        }
                    public_result = self._public_probe_result(result)
                    probe_results.append(public_result)
                    self._log(f"probe result {public_result}")
                    if result["read_ok"]:
                        with self.control_lock:
                            self.capture = result["capture"]
                            self.frame = result["frame"]
                            self.selected_index = candidate_index
                            self.selected_backend = backend_name
                            self.backend_name = backend_name
                            self.backend = BACKEND_MAP.get(backend_name, cv2.CAP_ANY)
                            self.last_frame_time = time.time()
                            self.last_error = ""
                            self.auto_focus_set_ok = result.get("autofocus_set_ok")
                            self.auto_focus_last_error = result.get("autofocus_error", "")
                            self.auto_focus_supported = bool(result.get("auto_focus_supported"))
                            self.focus_supported = bool(result.get("focus_supported"))
                            self.focus_value = result.get("focus_value")
                            self.focus_reported_value = result.get("focus_reported_value")
                            self.focus_min = result.get("focus_min")
                            self.focus_max = result.get("focus_max")
                            self.focus_step = result.get("focus_step")
                            self.focus_set_ok = result.get("focus_set_ok")
                            self.focus_last_error = result.get("focus_error", "")
                            self.probe_results = probe_results
                        self._log(
                            f"selected camera index={candidate_index} backend={backend_name} "
                            f"shape={result.get('shape')} autofocus_ok={result.get('autofocus_set_ok')} "
                            f"focus_supported={result.get('focus_supported')}"
                        )
                        return
                    capture = result.get("capture")
                    if capture is not None:
                        try:
                            capture.release()
                        except Exception:
                            pass
                    # 留一点间隔避免连续快速开关导致后端状态污染
                    time.sleep(0.05)

            self.probe_results = probe_results
            if probe_results:
                last_message = probe_results[-1]["error"]
                self.last_error = f"failed to find usable camera; last error: {last_message}"
            else:
                self.last_error = "failed to find usable camera; no probe candidates configured"
            self._log(f"open capture failed: {self.last_error}")
        except Exception as exc:
            self.last_error = f"open capture exception: {exc}"
            self.probe_results = []
            self.selected_index = None
            self.selected_backend = None
            self._log(f"open capture exception: {exc}", exc=exc)

    def _release_capture(self) -> None:
        if self.capture is not None:
            try:
                self._log(f"releasing camera capture index={self.selected_index} backend={self.selected_backend}")
                self.capture.release()
            except Exception as exc:
                self._log(f"capture release exception: {exc}", exc=exc)
        self.capture = None
        self.auto_focus_supported = False
        self.focus_supported = False
        self.focus_value = None
        self.focus_reported_value = None
        self.focus_min = None
        self.focus_max = None
        self.focus_step = None

    def _build_candidate_indices(self) -> list[int]:
        if self.selection_mode == "fixed":
            return [self.index]

        if self.selection_mode == "auto_prefer_external":
            preferred_first = list(self.preferred_indices) + list(self.probe_indices)
            return self._dedupe_indices(preferred_first)

        if self.selection_mode == "auto_first_available":
            return self._dedupe_indices(self.probe_indices)

        return [self.index]

    def _probe_camera(self, index: int, backend_name: str) -> dict[str, Any]:
        backend = BACKEND_MAP.get(backend_name, cv2.CAP_ANY)
        started_at = time.time()
        self._log(f"VideoCapture create begin index={index} backend={backend_name}")
        capture = cv2.VideoCapture(index, backend)
        open_elapsed_ms = int((time.time() - started_at) * 1000)
        self._log(
            f"VideoCapture create end index={index} backend={backend_name} "
            f"opened={bool(capture and capture.isOpened())} elapsed_ms={open_elapsed_ms}"
        )
        result: dict[str, Any] = {
            "index": index,
            "backend": backend_name,
            "opened": bool(capture and capture.isOpened()),
            "read_ok": False,
            "shape": None,
            "error": "",
            "autofocus_requested": self.auto_focus,
            "autofocus_set_ok": None,
            "autofocus_error": "",
            "auto_focus_supported": False,
            "focus_supported": False,
            "focus_value": None,
            "focus_reported_value": None,
            "focus_min": None,
            "focus_max": None,
            "focus_step": None,
            "focus_set_ok": None,
            "focus_error": "",
            "capture": capture,
            "frame": None,
        }

        if not capture or not capture.isOpened():
            result["error"] = f"open failed for index {index} with backend {backend_name}"
            return result

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        try:
            result["autofocus_set_ok"] = bool(capture.set(cv2.CAP_PROP_AUTOFOCUS, 1 if self.auto_focus else 0))
        except Exception as exc:
            result["autofocus_set_ok"] = False
            result["autofocus_error"] = str(exc)
        if self.manual_focus is not None and not self.auto_focus:
            try:
                result["focus_set_ok"] = bool(capture.set(cv2.CAP_PROP_FOCUS, self.manual_focus))
            except Exception as exc:
                result["focus_set_ok"] = False
                result["focus_error"] = str(exc)
        focus_state = self._read_focus_state(capture, result["focus_error"])
        result.update(focus_state)

        # 与 _reader_loop 一致：先 grab() 刷掉 DSHOW 内部缓冲的旧帧
        for _ in range(3):
            capture.grab()
        ok, frame = capture.retrieve()
        result["read_elapsed_ms"] = int((time.time() - started_at) * 1000)
        if not ok or frame is None:
            result["error"] = f"read failed for index {index} with backend {backend_name}"
            return result

        result["read_ok"] = True
        result["shape"] = tuple(frame.shape)
        result["frame"] = frame
        return result

    @staticmethod
    def _public_probe_result(result: dict[str, Any]) -> dict[str, Any]:
        return {
            "index": result["index"],
            "backend": result["backend"],
            "opened": result["opened"],
            "read_ok": result["read_ok"],
            "shape": result["shape"],
            "error": result["error"],
            "autofocus_requested": result.get("autofocus_requested"),
            "autofocus_set_ok": result.get("autofocus_set_ok"),
            "autofocus_error": result.get("autofocus_error", ""),
            "auto_focus_supported": result.get("auto_focus_supported"),
            "focus_supported": result.get("focus_supported"),
            "focus_value": result.get("focus_value"),
            "focus_reported_value": result.get("focus_reported_value"),
            "focus_min": result.get("focus_min"),
            "focus_max": result.get("focus_max"),
            "focus_step": result.get("focus_step"),
            "focus_set_ok": result.get("focus_set_ok"),
            "focus_error": result.get("focus_error", ""),
            "read_elapsed_ms": result.get("read_elapsed_ms"),
        }

    def _seconds_since_last_frame(self) -> float | None:
        if not self.last_frame_time:
            return None
        return max(time.time() - self.last_frame_time, 0.0)

    def _restart_if_frame_stale(self) -> None:
        if self.stale_frame_seconds <= 0 or not self.running:
            return
        if not self.capture or not self.capture.isOpened() or not self.last_frame_time:
            return
        age = time.time() - self.last_frame_time
        if age < self.stale_frame_seconds:
            return
        now = time.time()
        if now - self._last_stale_log_time >= self.stale_frame_seconds:
            self._last_stale_log_time = now
            self._log(
                f"camera frame stale age_seconds={age:.2f} threshold={self.stale_frame_seconds}; "
                "releasing capture for restart"
            )
        self.last_error = f"camera frame stale for {age:.1f}s"
        self._release_capture()

    def _log(self, message: str, *, exc: BaseException | None = None) -> None:
        if not self.log_enabled:
            return
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            thread_name = threading.current_thread().name
            lines = [f"[{timestamp}] [{thread_name}] {message}"]
            if exc is not None:
                lines.append("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip())
            with open(self.log_path, "a", encoding="utf-8") as file:
                file.write("\n".join(lines) + "\n")
        except Exception:
            pass

    def _refresh_focus_state(self, error: str = "") -> None:
        if not self.capture or not self.capture.isOpened():
            return
        state = self._read_focus_state(self.capture, error)
        self.auto_focus_supported = bool(state["auto_focus_supported"])
        self.focus_supported = bool(state["focus_supported"])
        self.focus_value = state["focus_value"]
        self.focus_reported_value = state["focus_reported_value"]
        self.focus_min = state["focus_min"]
        self.focus_max = state["focus_max"]
        self.focus_step = state["focus_step"]
        self.focus_last_error = state["focus_error"]

    def _read_focus_state(self, capture: cv2.VideoCapture, error: str = "") -> dict[str, Any]:
        auto_value = self._safe_get(capture, cv2.CAP_PROP_AUTOFOCUS)
        focus_value = self._safe_get(capture, cv2.CAP_PROP_FOCUS)
        focus_supported = focus_value is not None and focus_value >= 0
        auto_supported = (
            (auto_value is not None and auto_value >= 0)
            or self.auto_focus_set_ok is True
        )
        focus_min = min(self.focus_probe_values) if focus_supported and self.focus_probe_values else None
        focus_max = max(self.focus_probe_values) if focus_supported and self.focus_probe_values else None
        step = None
        if focus_supported and len(self.focus_probe_values) > 1:
            ordered = sorted(set(self.focus_probe_values))
            deltas = [b - a for a, b in zip(ordered, ordered[1:]) if b > a]
            step = min(deltas) if deltas else 1
        if not focus_supported and not error:
            error = "manual focus is not exposed by this camera driver"
        display_focus = focus_value if focus_supported else None
        if focus_supported and not self.auto_focus and self.manual_focus is not None:
            display_focus = float(self.manual_focus)
        return {
            "auto_focus_supported": auto_supported,
            "focus_supported": focus_supported,
            "focus_value": display_focus,
            "focus_reported_value": focus_value if focus_supported else None,
            "focus_min": focus_min,
            "focus_max": focus_max,
            "focus_step": step,
            "focus_error": error,
        }

    @staticmethod
    def _safe_get(capture: cv2.VideoCapture, prop: int) -> float | None:
        try:
            value = float(capture.get(prop))
        except Exception:
            return None
        return value

    @staticmethod
    def _dedupe_indices(values: list[int]) -> list[int]:
        ordered: list[int] = []
        seen: set[int] = set()
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            ordered.append(value)
        return ordered

    @staticmethod
    def _dedupe_backends(values: list[str]) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for value in values:
            key = value.upper()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(key)
        return ordered
