from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2

from camera_driver import CameraDriver, FrameUnavailableError
from config_manager import load_config
from runtime_paths import get_app_paths
from video_recorder import VideoRecorder


_STARTUP_FRAME_WAIT_SECONDS = 5


@dataclass(frozen=True)
class BurstCaptureConfig:
    frame_count: int = 16
    fps: int = 16
    min_valid_frames: int = 12
    resolution: tuple[int, int] = (1280, 720)
    codec: str = "mp4v"
    output_frame_indices: tuple[int, ...] = (3, 7, 10, 13)


@dataclass
class BurstCaptureManifest:
    take_dir: Path
    scenario: str
    take: str
    status: str
    frame_paths: list[Path]
    valid_frame_count: int
    output_frame_indices: tuple[int, ...]
    video_path: Path | None
    error: str | None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "take_dir": str(self.take_dir),
            "scenario": self.scenario,
            "take": self.take,
            "status": self.status,
            "frame_paths": [str(path) for path in self.frame_paths],
            "valid_frame_count": self.valid_frame_count,
            "output_frame_indices": list(self.output_frame_indices),
            "video_path": str(self.video_path) if self.video_path is not None else None,
            "error": self.error,
        }


def capture_take(
    camera: Any,
    output_root: Path,
    scenario: str,
    take: str,
    config: BurstCaptureConfig,
    *,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> BurstCaptureManifest:
    take_dir = Path(output_root) / scenario / take
    frames_dir = take_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = take_dir / "burst.avi"
    video_path: Path | None = None
    frame_paths: list[Path] = []
    error: str | None = None
    fatal_error = False
    recorder = VideoRecorder(
        fps=config.fps,
        resolution=config.resolution,
        codec=config.codec,
    )
    try:
        recorder.start(output_video_path)
        video_path = output_video_path
        first_frame = None
        startup_attempts = max(config.fps * _STARTUP_FRAME_WAIT_SECONDS, 1)
        for attempt in range(startup_attempts):
            try:
                first_frame = camera.get_frame()
                error = None
                break
            except FrameUnavailableError as exc:
                error = str(exc)
                if attempt < startup_attempts - 1:
                    sleep_fn(1 / max(config.fps, 1))

        if first_frame is not None:
            for index in range(1, config.frame_count + 1):
                if index == 1:
                    frame = first_frame
                else:
                    try:
                        frame = camera.get_frame()
                    except Exception as exc:
                        error = str(exc)
                        break
                frame_path = frames_dir / f"{index:06d}.jpg"
                if not cv2.imwrite(str(frame_path), frame):
                    error = f"failed to write frame: {frame_path}"
                    fatal_error = True
                    break
                frame_paths.append(frame_path)
                try:
                    recorder.write_frame(frame)
                except Exception as exc:
                    error = str(exc)
                    fatal_error = True
                    break
                if index < config.frame_count:
                    sleep_fn(1 / config.fps)
    except Exception as exc:
        error = str(exc)
        fatal_error = True
    finally:
        try:
            recorder.stop()
        except Exception as exc:
            fatal_error = True
            if error is None:
                error = str(exc)

    valid_frame_count = len(frame_paths)
    missing_output_indices = [
        index
        for index in config.output_frame_indices
        if index < 1 or index > valid_frame_count
    ]
    status = (
        "ok"
        if not fatal_error
        and valid_frame_count >= config.min_valid_frames
        and not missing_output_indices
        else "failed"
    )
    if status == "failed" and error is None and missing_output_indices:
        missing = ", ".join(str(index) for index in missing_output_indices)
        error = f"missing required output frames: {missing}"
    manifest = BurstCaptureManifest(
        take_dir=take_dir,
        scenario=scenario,
        take=take,
        status=status,
        frame_paths=frame_paths,
        valid_frame_count=valid_frame_count,
        output_frame_indices=config.output_frame_indices,
        video_path=video_path,
        error=error,
    )
    (take_dir / "metadata.json").write_text(
        json.dumps(manifest.to_json_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


async def _run_capture_cli(
    camera_config: dict[str, Any],
    output_root: Path,
    scenario: str,
    take: str,
    config: BurstCaptureConfig,
    *,
    driver_factory: Callable[[dict[str, Any]], Any] = CameraDriver,
    capture_fn: Callable[..., BurstCaptureManifest] = capture_take,
) -> BurstCaptureManifest:
    driver = driver_factory(camera_config)
    driver.start()
    try:
        return await asyncio.to_thread(
            capture_fn,
            driver,
            output_root,
            scenario,
            take,
            config,
        )
    finally:
        driver.stop()


def _default_output_root() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return get_app_paths()["output_dir"] / "burst_eval_inputs" / timestamp


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture an offline evaluation burst.")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--take", required=True)
    parser.add_argument("--frame-count", type=int, default=16)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = BurstCaptureConfig(frame_count=args.frame_count, fps=args.fps)
    output_root = args.output_root or _default_output_root()
    app_config = load_config()
    manifest = asyncio.run(
        _run_capture_cli(
            app_config["camera"],
            output_root,
            args.scenario,
            args.take,
            config,
        )
    )
    print(json.dumps(manifest.to_json_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
