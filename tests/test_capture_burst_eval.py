import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

import capture_burst_eval as burst_eval
from camera_driver import FrameUnavailableError
from capture_burst_eval import (
    BurstCaptureConfig,
    BurstCaptureManifest,
    _run_capture_cli,
    capture_take,
)


class FakeCamera:
    def __init__(self, available_frames: int) -> None:
        self.available_frames = available_frames
        self.read_count = 0

    def get_frame(self) -> np.ndarray:
        if self.read_count >= self.available_frames:
            raise RuntimeError("camera frame stream ended")
        self.read_count += 1
        return np.full((48, 64, 3), self.read_count, dtype=np.uint8)


class DelayedReadyCamera(FakeCamera):
    def __init__(self, available_frames: int, unavailable_reads: int) -> None:
        super().__init__(available_frames)
        self.unavailable_reads = unavailable_reads
        self.attempt_count = 0

    def get_frame(self) -> np.ndarray:
        self.attempt_count += 1
        if self.attempt_count <= self.unavailable_reads:
            raise FrameUnavailableError("camera frame is not available yet")
        return super().get_frame()


class FailingRecorder:
    def __init__(self, fail_on: str) -> None:
        self.fail_on = fail_on
        self.stop_called = False

    def start(self, output_path: Path) -> None:
        if self.fail_on == "start":
            raise RuntimeError("recorder start failed")

    def write_frame(self, frame: np.ndarray) -> None:
        if self.fail_on == "write":
            raise RuntimeError("recorder write failed")

    def stop(self) -> Path:
        self.stop_called = True
        return Path("stopped.avi")


def _config(**changes: object) -> BurstCaptureConfig:
    values = {
        "frame_count": 16,
        "fps": 16,
        "min_valid_frames": 12,
        "resolution": (64, 48),
    }
    values.update(changes)
    return BurstCaptureConfig(**values)


def test_burst_capture_config_defaults() -> None:
    config = BurstCaptureConfig()

    assert config.frame_count == 16
    assert config.fps == 16
    assert config.min_valid_frames == 12
    assert config.resolution == (1280, 720)
    assert config.codec == "mp4v"
    assert config.output_frame_indices == (3, 7, 10, 13)


def test_capture_take_writes_sixteen_frames_video_and_metadata(tmp_path: Path) -> None:
    manifest = capture_take(
        FakeCamera(available_frames=16),
        tmp_path,
        "visitor_close_contact",
        "take_01",
        _config(),
        sleep_fn=lambda _: None,
    )
    take_dir = tmp_path / "visitor_close_contact" / "take_01"
    metadata = json.loads((take_dir / "metadata.json").read_text(encoding="utf-8"))

    assert manifest.status == "ok"
    assert manifest.valid_frame_count == 16
    assert len(manifest.frame_paths) == 16
    assert [path.name for path in manifest.frame_paths] == [
        f"{index:06d}.jpg" for index in range(1, 17)
    ]
    assert all(path.exists() for path in manifest.frame_paths)
    assert manifest.video_path == take_dir / "burst.avi"
    assert manifest.video_path.exists()
    assert manifest.video_path.stat().st_size > 0
    assert metadata["status"] == "ok"
    assert metadata["valid_frame_count"] == 16
    assert metadata["take_dir"] == str(take_dir)
    assert metadata["video_path"] == str(take_dir / "burst.avi")
    assert metadata["frame_paths"] == [str(path) for path in manifest.frame_paths]
    assert metadata["output_frame_indices"] == [3, 7, 10, 13]


def test_capture_take_records_failed_metadata_when_only_eight_frames_are_read(
    tmp_path: Path,
) -> None:
    manifest = capture_take(
        FakeCamera(available_frames=8),
        tmp_path,
        "visitor_close_contact",
        "take_02",
        _config(),
        sleep_fn=lambda _: None,
    )
    metadata_path = tmp_path / "visitor_close_contact" / "take_02" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert manifest.status == "failed"
    assert manifest.valid_frame_count == 8
    assert len(manifest.frame_paths) == 8
    assert manifest.error == "camera frame stream ended"
    assert metadata["status"] == "failed"
    assert metadata["valid_frame_count"] == 8
    assert metadata["error"] == "camera frame stream ended"
    assert metadata_path.exists()


def test_capture_take_waits_for_first_frame_while_camera_becomes_ready(
    tmp_path: Path,
) -> None:
    camera = DelayedReadyCamera(available_frames=4, unavailable_reads=2)

    manifest = capture_take(
        camera,
        tmp_path,
        "visitor_close_contact",
        "take_delayed_ready",
        _config(
            frame_count=4,
            min_valid_frames=4,
            output_frame_indices=(1, 4),
        ),
        sleep_fn=lambda _: None,
    )

    assert manifest.status == "ok"
    assert manifest.valid_frame_count == 4
    assert camera.attempt_count == 6


def test_capture_take_fails_when_minimum_count_does_not_include_output_frames(
    tmp_path: Path,
) -> None:
    manifest = capture_take(
        FakeCamera(available_frames=12),
        tmp_path,
        "visitor_close_contact",
        "take_missing_output",
        _config(frame_count=12),
        sleep_fn=lambda _: None,
    )

    assert manifest.valid_frame_count == 12
    assert manifest.status == "failed"


def test_capture_take_accepts_custom_output_index_at_last_available_frame(
    tmp_path: Path,
) -> None:
    manifest = capture_take(
        FakeCamera(available_frames=12),
        tmp_path,
        "visitor_close_contact",
        "take_output_boundary",
        _config(frame_count=12, output_frame_indices=(3, 7, 10, 12)),
        sleep_fn=lambda _: None,
    )

    assert manifest.valid_frame_count == 12
    assert manifest.status == "ok"


@pytest.mark.parametrize(
    ("fail_on", "expected_error"),
    [
        ("start", "recorder start failed"),
        ("write", "recorder write failed"),
    ],
)
def test_capture_take_stops_failed_recorder_and_writes_failed_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_on: str,
    expected_error: str,
) -> None:
    recorder = FailingRecorder(fail_on)
    monkeypatch.setattr(burst_eval, "VideoRecorder", lambda **_: recorder)

    manifest = capture_take(
        FakeCamera(available_frames=16),
        tmp_path,
        "visitor_close_contact",
        f"take_recorder_{fail_on}",
        _config(),
        sleep_fn=lambda _: None,
    )
    metadata = json.loads(
        (manifest.take_dir / "metadata.json").read_text(encoding="utf-8")
    )

    assert recorder.stop_called
    assert manifest.status == "failed"
    assert manifest.error == expected_error
    assert metadata["status"] == "failed"
    assert metadata["error"] == expected_error


def test_capture_take_reports_frame_written_before_video_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = FailingRecorder("write")
    monkeypatch.setattr(burst_eval, "VideoRecorder", lambda **_: recorder)

    manifest = capture_take(
        FakeCamera(available_frames=16),
        tmp_path,
        "visitor_close_contact",
        "take_recorder_write_consistency",
        _config(),
        sleep_fn=lambda _: None,
    )
    metadata = json.loads(
        (manifest.take_dir / "metadata.json").read_text(encoding="utf-8")
    )
    disk_frame_paths = sorted((manifest.take_dir / "frames").glob("*.jpg"))

    assert disk_frame_paths == manifest.frame_paths
    assert metadata["frame_paths"] == [str(path) for path in disk_frame_paths]
    assert manifest.valid_frame_count == len(disk_frame_paths) == 1
    assert metadata["valid_frame_count"] == len(disk_frame_paths)


def test_capture_take_omits_video_path_when_recorder_start_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = FailingRecorder("start")
    monkeypatch.setattr(burst_eval, "VideoRecorder", lambda **_: recorder)

    manifest = capture_take(
        FakeCamera(available_frames=16),
        tmp_path,
        "visitor_close_contact",
        "take_recorder_start_without_video",
        _config(),
        sleep_fn=lambda _: None,
    )
    metadata = json.loads(
        (manifest.take_dir / "metadata.json").read_text(encoding="utf-8")
    )

    assert manifest.video_path is None
    assert metadata["video_path"] is None


def test_run_capture_cli_starts_driver_inside_running_loop_and_always_stops(
    tmp_path: Path,
) -> None:
    lifecycle: list[str] = []

    class LoopCheckingCamera:
        def __init__(self, camera_config: dict[str, object]) -> None:
            assert camera_config == {"index": 2}

        def start(self) -> None:
            asyncio.get_running_loop()
            lifecycle.append("start")

        def stop(self) -> None:
            lifecycle.append("stop")

    expected = BurstCaptureManifest(
        take_dir=tmp_path,
        scenario="visitor_close_contact",
        take="take_03",
        status="failed",
        frame_paths=[],
        valid_frame_count=0,
        output_frame_indices=(3, 7, 10, 13),
        video_path=tmp_path / "burst.avi",
        error="stub",
    )

    def fake_capture(
        camera: LoopCheckingCamera,
        output_root: Path,
        scenario: str,
        take: str,
        config: BurstCaptureConfig,
    ) -> BurstCaptureManifest:
        assert lifecycle == ["start"]
        assert output_root == tmp_path
        assert scenario == "visitor_close_contact"
        assert take == "take_03"
        assert config == BurstCaptureConfig()
        return expected

    result = asyncio.run(
        _run_capture_cli(
            {"index": 2},
            tmp_path,
            "visitor_close_contact",
            "take_03",
            BurstCaptureConfig(),
            driver_factory=LoopCheckingCamera,
            capture_fn=fake_capture,
        )
    )

    assert result is expected
    assert lifecycle == ["start", "stop"]
