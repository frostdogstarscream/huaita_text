import threading
import time
from typing import Any, Generator

import cv2
import numpy as np


class CameraUnavailableError(RuntimeError):
    pass


class FrameUnavailableError(RuntimeError):
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
        self.jpeg_quality = int(camera_config["jpeg_quality"])
        self.auto_focus_set_ok: bool | None = None
        self.auto_focus_last_error = ""

        self.capture: cv2.VideoCapture | None = None
        self.frame: np.ndarray | None = None
        self.frame_lock = threading.Lock()
        self.worker: threading.Thread | None = None
        self.running = False
        self.last_error = "camera not started"
        self.last_frame_time = 0.0
        self.selected_index: int | None = None
        self.selected_backend: str | None = None
        self.probe_attempts = 0
        self.probe_results: list[dict[str, Any]] = []

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.worker = threading.Thread(target=self._reader_loop, daemon=True)
        self.worker.start()

    def stop(self) -> None:
        self.running = False
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=2)
        self._release_capture()

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
            "last_error": self.last_error,
            "has_frame": self.frame is not None,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "auto_focus": self.auto_focus,
            "auto_focus_set_ok": self.auto_focus_set_ok,
            "auto_focus_last_error": self.auto_focus_last_error,
        }

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

    def _reader_loop(self) -> None:
        while self.running:
            if not self.capture or not self.capture.isOpened():
                self._open_capture()
                if not self.capture or not self.capture.isOpened():
                    time.sleep(1.0)
                    continue

            ok, frame = self.capture.read()
            if not ok or frame is None:
                self.last_error = "read failed"
                self._release_capture()
                time.sleep(0.5)
                continue

            with self.frame_lock:
                self.frame = frame
                self.last_frame_time = time.time()
                self.last_error = ""

    def _open_capture(self) -> None:
        self._release_capture()
        self.probe_attempts += 1
        self.selected_index = None
        self.selected_backend = None
        candidates = self._build_candidate_indices()
        probe_results: list[dict[str, Any]] = []

        for candidate_index in candidates:
            for backend_name in self.backend_order:
                result = self._probe_camera(candidate_index, backend_name)
                probe_results.append(self._public_probe_result(result))
                if result["read_ok"]:
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
                    self.probe_results = probe_results
                    return
                capture = result.get("capture")
                if capture is not None:
                    capture.release()

        self.probe_results = probe_results
        if probe_results:
            last_message = probe_results[-1]["error"]
            self.last_error = f"failed to find usable camera; last error: {last_message}"
        else:
            self.last_error = "failed to find usable camera; no probe candidates configured"

    def _release_capture(self) -> None:
        if self.capture is not None:
            self.capture.release()
        self.capture = None

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
        capture = cv2.VideoCapture(index, backend)
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

        ok, frame = capture.read()
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
        }

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
