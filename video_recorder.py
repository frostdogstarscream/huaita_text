from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


class VideoRecorderError(RuntimeError):
    pass


class VideoRecorder:
    def __init__(
        self,
        *,
        fps: int = 20,
        resolution: tuple[int, int] = (1280, 720),
        codec: str = "mp4v",
        buffer_duration_s: float = 2.0,
        logger: Any | None = None,
    ) -> None:
        self._fps = fps
        self._resolution = resolution
        self._codec = codec
        self._buffer_duration_s = buffer_duration_s
        self._logger = logger or print

        self._writer: cv2.VideoWriter | None = None
        self._output_path: Path | None = None
        self._frame_buffer: deque[tuple[float, np.ndarray]] = deque()
        self._buffer_size = max(int(fps * buffer_duration_s), 1)
        self._started_at: float | None = None
        self._frame_count = 0

    def start(self, output_path: Path) -> None:
        if self._writer is not None:
            raise VideoRecorderError("Recording already in progress.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*self._codec)
        self._writer = cv2.VideoWriter(
            str(output_path), fourcc, self._fps, self._resolution
        )
        if not self._writer.isOpened():
            raise VideoRecorderError(f"Failed to open video writer: {output_path}")
        self._output_path = output_path
        self._started_at = time.perf_counter()
        self._frame_count = 0
        self._frame_buffer.clear()
        self._logger(f"[VideoRecorder] started: {output_path}")

    def write_frame(self, frame: np.ndarray) -> None:
        if self._writer is None:
            raise VideoRecorderError("Recording not started.")
        if frame.shape[:2] != (self._resolution[1], self._resolution[0]):
            frame = cv2.resize(frame, self._resolution)
        self._writer.write(frame)
        elapsed = time.perf_counter() - self._started_at
        self._frame_buffer.append((elapsed, frame.copy()))
        while len(self._frame_buffer) > self._buffer_size:
            self._frame_buffer.popleft()
        self._frame_count += 1

    def stop(self) -> Path:
        if self._writer is None:
            raise VideoRecorderError("Recording not started.")
        self._writer.release()
        self._writer = None
        self._logger(
            f"[VideoRecorder] stopped: {self._output_path} "
            f"({self._frame_count} frames, "
            f"{len(self._frame_buffer)} buffered)"
        )
        return self._output_path  # type: ignore[return-value]

    @property
    def is_recording(self) -> bool:
        return self._writer is not None

    def extract_frame_at(self, timestamp: float) -> Image.Image:
        """Extract the frame closest to *timestamp* (seconds relative to start)."""
        if not self._frame_buffer:
            raise VideoRecorderError("No frames buffered.")
        best_idx = 0
        best_dist = float("inf")
        for i, (t, _frame) in enumerate(self._frame_buffer):
            dist = abs(t - timestamp)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        _, frame = self._frame_buffer[best_idx]
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def extract_frames(
        self, timestamps: list[float], window_start: float
    ) -> list[Image.Image]:
        """Extract frames at absolute timestamps relative to *window_start*.

        Each element of *timestamps* is an offset in seconds from
        *window_start*.  The call finds the buffered frame closest to
        ``window_start + offset`` for each offset.
        """
        return [self.extract_frame_at(window_start + offset) for offset in timestamps]
