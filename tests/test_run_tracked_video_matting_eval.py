import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from run_tracked_video_matting_eval import (
    aggregate_metric_rows,
    apply_tracked_alpha_constraints,
    compute_edge_temporal_jitter,
    compute_tracked_metrics,
    run_session,
    run_take,
    select_output_frames,
)
from subject_edge_refine import SubjectEdgeRefineConfig
from subject_instance_tracking import (
    TrackedCandidate,
    TrackedFrameResult,
    TrackedInstanceSequence,
)


def _tracked_frame() -> TrackedFrameResult:
    subject = np.zeros((20, 20), dtype=bool)
    subject[4:18, 3:14] = True
    visitor = np.zeros((20, 20), dtype=bool)
    visitor[6:16, 15:19] = True
    core = np.zeros((20, 20), dtype=bool)
    core[6:16, 5:12] = True
    sure_background = ~subject | visitor
    unknown = subject & ~core
    return TrackedFrameResult(
        frame_path=Path("frame.jpg"),
        selected=TrackedCandidate(7, (3, 4, 14, 18), 0.95, subject),
        visitors=[TrackedCandidate(9, (15, 6, 19, 16), 0.9, visitor)],
        trimap=np.where(core, 255, np.where(unknown, 128, 0)).astype(np.uint8),
        sure_foreground=core,
        sure_background=sure_background,
        unknown=unknown,
    )


def test_select_output_frames_uses_configured_four_indices() -> None:
    selected = select_output_frames(list(range(16)), [3, 7, 10, 13])

    assert selected == [3, 7, 10, 13]


def test_select_output_frames_requires_four_valid_indices() -> None:
    with pytest.raises(ValueError):
        select_output_frames(list(range(4)), [0, 1, 4, 3])
    with pytest.raises(ValueError):
        select_output_frames(list(range(4)), [0, 1])


def test_apply_tracked_constraints_clears_visitors_and_preserves_subject_core() -> None:
    frame = _tracked_frame()
    raw = np.full((20, 20), 120, dtype=np.uint8)
    image = apply_tracked_alpha_constraints(Image.new("RGB", (20, 20), "white"), raw, frame)
    alpha = np.array(image.getchannel("A"))

    assert np.all(alpha[frame.sure_background] == 0)
    assert np.all(alpha[frame.sure_foreground] == 255)
    assert np.all(alpha[frame.unknown] == 120)


def test_tracked_metrics_report_visitor_core_outside_and_edge_jitter() -> None:
    frame = _tracked_frame()
    raw = np.full((20, 20), 120, dtype=np.uint8)
    image = apply_tracked_alpha_constraints(Image.new("RGB", (20, 20), "white"), raw, frame)

    metrics = compute_tracked_metrics(image, frame)

    assert metrics["visitor_track_alpha_ratio"] == 0
    assert metrics["subject_core_missing_ratio"] == 0
    assert metrics["outside_subject_soft_alpha_ratio"] == 0
    assert metrics["foreground_px"] > 0
    assert compute_edge_temporal_jitter([image, image], [frame, frame]) == 0


def test_metric_aggregation_reports_means_maxima_and_elapsed() -> None:
    summary = aggregate_metric_rows(
        [
            {"visitor_track_alpha_ratio": 0.0, "foreground_px": 100.0},
            {"visitor_track_alpha_ratio": 0.2, "foreground_px": 120.0},
        ],
        elapsed_seconds=1.25,
    )

    assert summary["elapsed_seconds"] == 1.25
    assert summary["visitor_track_alpha_ratio_avg"] == pytest.approx(0.1)
    assert summary["visitor_track_alpha_ratio_max"] == pytest.approx(0.2)
    assert summary["foreground_px_avg"] == pytest.approx(110.0)


def _create_fake_take(tmp_path: Path, frame_count: int = 16) -> Path:
    take_dir = tmp_path / "take"
    frames_dir = take_dir / "frames"
    frames_dir.mkdir(parents=True)
    frame_paths = []
    for index in range(frame_count):
        path = frames_dir / f"{index + 1:06d}.jpg"
        Image.new("RGB", (20, 20), (30 + index, 20, 10)).save(path)
        frame_paths.append(str(path))
    video_path = take_dir / "burst.avi"
    video_path.write_bytes(b"fake-video")
    manifest = {
        "take_dir": str(take_dir),
        "scenario": "visitor_close_contact",
        "take": "take_01",
        "status": "ok",
        "frame_paths": frame_paths,
        "valid_frame_count": frame_count,
        "output_frame_indices": [3, 7, 10, 13],
        "video_path": str(video_path),
        "error": None,
    }
    manifest_path = take_dir / "metadata.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _ok_sequence(frame_count: int) -> TrackedInstanceSequence:
    frames = []
    for index in range(frame_count):
        frame = _tracked_frame()
        frames.append(
            TrackedFrameResult(
                frame_path=Path(f"frame_{index}.jpg"),
                selected=frame.selected,
                visitors=frame.visitors,
                trimap=frame.trimap,
                sure_foreground=frame.sure_foreground,
                sure_background=frame.sure_background,
                unknown=frame.unknown,
            )
        )
    return TrackedInstanceSequence(7, frames, 0, 0, "ok")


def _contact_sequence(frame_count: int) -> TrackedInstanceSequence:
    subject = np.zeros((20, 20), dtype=bool)
    subject[4:18, 3:14] = True
    visitor = np.zeros((20, 20), dtype=bool)
    visitor[6:16, 12:19] = True
    core = np.zeros((20, 20), dtype=bool)
    core[7:15, 6:11] = True
    frame = TrackedFrameResult(
        frame_path=Path("frame.jpg"),
        selected=TrackedCandidate(7, (3, 4, 14, 18), 0.95, subject),
        visitors=[TrackedCandidate(9, (12, 6, 19, 16), 0.9, visitor)],
        trimap=np.where(core, 255, 0).astype(np.uint8),
        sure_foreground=core,
        sure_background=(~subject) | visitor,
        unknown=subject & ~core & ~visitor,
    )
    return TrackedInstanceSequence(7, [frame for _ in range(frame_count)], 0, 0, "ok")


class FakeSequenceTracker:
    def __init__(self, sequence: TrackedInstanceSequence) -> None:
        self.sequence = sequence
        self.calls = 0
        self.reset_calls = 0

    def track_paths(self, frame_paths: list[Path]) -> TrackedInstanceSequence:
        self.calls += 1
        assert len(frame_paths) == 16
        return self.sequence

    def reset(self) -> None:
        self.reset_calls += 1


class FakeVideoMatting:
    def __init__(self, alpha_frames: list[Image.Image]) -> None:
        self.alpha_frames = alpha_frames
        self.calls = 0

    def process_video(self, _video_path: Path, first_mask: np.ndarray, _output_dir: Path, *, save_frames: bool = True):
        self.calls += 1
        assert save_frames is True
        assert np.count_nonzero(first_mask) > 0
        return self.alpha_frames


class FakeModnet:
    def matte_image_file(self, source_path: Path, _instance: object) -> SimpleNamespace:
        image = Image.open(source_path).convert("RGBA")
        alpha = np.full((image.height, image.width), 120, dtype=np.uint8)
        image.putalpha(Image.fromarray(alpha, "L"))
        return SimpleNamespace(image=image)


class FailingModnet:
    def matte_image_file(self, _source_path: Path, _instance: object) -> SimpleNamespace:
        raise RuntimeError("modnet unavailable")


def test_run_take_generates_three_branches_and_four_final_outputs(tmp_path: Path) -> None:
    manifest_path = _create_fake_take(tmp_path)
    output_dir = tmp_path / "out"
    alpha_frames = [Image.fromarray(np.full((20, 20), 120, dtype=np.uint8), "L") for _ in range(16)]

    summary = run_take(
        manifest_path,
        output_dir,
        tracker=FakeSequenceTracker(_ok_sequence(16)),
        matanyone=FakeVideoMatting(alpha_frames),
        modnet=FakeModnet(),
        compose_final=lambda image, *_: image,
    )

    assert summary["status"] == "ok"
    for branch in (
        "tracked_yolo_seg_mask",
        "tracked_modnet_4frame",
        "tracked_matanyone_current",
        "tracked_matanyone_subject_priority",
        "tracked_matanyone_body_refine",
    ):
        assert len(list((output_dir / branch / "final").glob("*.jpg"))) == 4
        assert branch in summary["branches"]
    assert "subject_contact_missing_ratio_avg" in summary["branches"]["tracked_matanyone_subject_priority"]["aggregate_metrics"]
    assert "body_outside_soft_alpha_ratio_avg" in summary["branches"]["tracked_matanyone_body_refine"]["aggregate_metrics"]
    assert (output_dir / "sheets" / "final_sheet.jpg").exists()
    assert (output_dir / "sheets" / "cutout_sheet.jpg").exists()
    assert (output_dir / "summary_metrics.json").exists()


def test_run_take_records_tracking_failure_without_invoking_matanyone(tmp_path: Path) -> None:
    manifest_path = _create_fake_take(tmp_path)
    failed = TrackedInstanceSequence(None, [], 0, 0, "tracking_failed", "no_primary_subject")
    matanyone = FakeVideoMatting([])

    summary = run_take(
        manifest_path,
        tmp_path / "out",
        tracker=FakeSequenceTracker(failed),
        matanyone=matanyone,
        modnet=None,
        compose_final=lambda image, *_: image,
    )

    assert summary["status"] == "tracking_failed"
    assert matanyone.calls == 0


def test_subject_priority_branch_restores_contact_deleted_by_current_rule(tmp_path: Path) -> None:
    manifest_path = _create_fake_take(tmp_path)
    output_dir = tmp_path / "out"
    alpha_frames = [Image.fromarray(np.full((20, 20), 120, dtype=np.uint8), "L") for _ in range(16)]

    summary = run_take(
        manifest_path,
        output_dir,
        tracker=FakeSequenceTracker(_contact_sequence(16)),
        matanyone=FakeVideoMatting(alpha_frames),
        modnet=None,
        compose_final=lambda image, *_: image,
        edge_config=SubjectEdgeRefineConfig(enabled=False),
    )

    current = summary["branches"]["tracked_matanyone_current"]["aggregate_metrics"]
    priority = summary["branches"]["tracked_matanyone_subject_priority"]["aggregate_metrics"]
    assert current["subject_contact_missing_ratio_avg"] > 0
    assert priority["subject_contact_missing_ratio_avg"] == 0
    assert priority["visitor_visible_residual_ratio_avg"] == 0
    assert (output_dir / "tracked_matanyone_subject_priority" / "debug" / "01_contact_conflict.png").exists()


def test_subject_priority_debug_uses_each_output_frames_own_contact_mask(tmp_path: Path) -> None:
    manifest_path = _create_fake_take(tmp_path)
    output_dir = tmp_path / "out"
    sequence = _contact_sequence(16)
    original = _ok_sequence(16).frames[7]
    no_contact = TrackedFrameResult(
        frame_path=original.frame_path,
        selected=original.selected,
        visitors=[],
        trimap=original.trimap,
        sure_foreground=original.sure_foreground,
        sure_background=~original.selected.mask,
        unknown=original.unknown,
    )
    frames = list(sequence.frames)
    frames[7] = no_contact
    mixed_sequence = TrackedInstanceSequence(7, frames, 0, 0, "ok")
    alpha_frames = [Image.fromarray(np.full((20, 20), 120, dtype=np.uint8), "L") for _ in range(16)]

    summary = run_take(
        manifest_path,
        output_dir,
        tracker=FakeSequenceTracker(mixed_sequence),
        matanyone=FakeVideoMatting(alpha_frames),
        modnet=None,
        compose_final=lambda image, *_: image,
        edge_config=SubjectEdgeRefineConfig(enabled=False),
    )

    rows = summary["branches"]["tracked_matanyone_subject_priority"]["frame_metrics"]
    assert rows[0]["contact_conflict_px"] > 0
    assert rows[1]["contact_conflict_px"] == 0


def test_body_refine_branch_writes_region_debug_outputs(tmp_path: Path) -> None:
    manifest_path = _create_fake_take(tmp_path)
    output_dir = tmp_path / "out"
    alpha_frames = [Image.fromarray(np.full((20, 20), 120, dtype=np.uint8), "L") for _ in range(16)]

    summary = run_take(
        manifest_path,
        output_dir,
        tracker=FakeSequenceTracker(_contact_sequence(16)),
        matanyone=FakeVideoMatting(alpha_frames),
        modnet=None,
        compose_final=lambda image, *_: image,
        edge_config=SubjectEdgeRefineConfig(enabled=False),
    )

    branch = summary["branches"]["tracked_matanyone_body_refine"]
    assert branch["status"] == "ok"
    assert "body_outside_soft_alpha_ratio_avg" in branch["aggregate_metrics"]
    debug_root = output_dir / "tracked_matanyone_body_refine" / "debug"
    assert (debug_root / "01_torso_region.png").exists()
    assert (debug_root / "01_arm_region.png").exists()
    assert (debug_root / "01_body_inner_rejudge.png").exists()
    assert (debug_root / "01_body_outer_support.png").exists()


def test_run_take_records_optional_branch_failure_and_keeps_mask_outputs(tmp_path: Path) -> None:
    manifest_path = _create_fake_take(tmp_path)
    output_dir = tmp_path / "out"

    summary = run_take(
        manifest_path,
        output_dir,
        tracker=FakeSequenceTracker(_ok_sequence(16)),
        matanyone=None,
        modnet=FailingModnet(),
        compose_final=lambda image, *_: image,
    )

    assert summary["status"] == "ok"
    assert summary["branches"]["tracked_modnet_4frame"]["status"] == "failed"
    assert "modnet unavailable" in summary["branches"]["tracked_modnet_4frame"]["error"]
    assert len(list((output_dir / "tracked_yolo_seg_mask" / "final").glob("*.jpg"))) == 4


def test_run_session_scans_take_manifests_and_writes_session_summary(tmp_path: Path) -> None:
    first = _create_fake_take(tmp_path / "first")
    second = _create_fake_take(tmp_path / "second")
    assert first.exists() and second.exists()
    output_dir = tmp_path / "session_out"

    tracker = FakeSequenceTracker(_ok_sequence(16))
    summary = run_session(
        tmp_path,
        output_dir,
        tracker=tracker,
        matanyone=None,
        modnet=None,
        compose_final=lambda image, *_: image,
    )

    assert summary["take_count"] == 2
    assert len(summary["takes"]) == 2
    assert tracker.reset_calls == 2
    assert (output_dir / "session_summary.json").exists()
