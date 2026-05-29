from pathlib import Path

import numpy as np
from PIL import Image

from subject_instance_segmentation import InstanceSegmentationConfig
from subject_instance_tracking import (
    SubjectInstanceTracker,
    TrackingConfig,
    TrackedCandidate,
)


def _mask(box: tuple[int, int, int, int], size: tuple[int, int] = (100, 100)) -> np.ndarray:
    result = np.zeros((size[1], size[0]), dtype=bool)
    left, top, right, bottom = box
    result[top:bottom, left:right] = True
    return result


def _candidate(
    track_id: int | None,
    box: tuple[int, int, int, int],
    confidence: float = 0.9,
) -> TrackedCandidate:
    return TrackedCandidate(
        track_id=track_id,
        bbox=tuple(float(value) for value in box),
        confidence=confidence,
        mask=_mask(box),
    )


def _frame_paths(tmp_path: Path, count: int) -> list[Path]:
    paths = []
    for index in range(count):
        path = tmp_path / f"frame_{index}.jpg"
        Image.new("RGB", (100, 100), (index, 0, 0)).save(path)
        paths.append(path)
    return paths


class FakeTrackBackend:
    def __init__(self, frames: list[list[TrackedCandidate]]) -> None:
        self._frames = list(frames)
        self.calls = 0

    def track(self, _frame_path: Path, _image_size: tuple[int, int]) -> list[TrackedCandidate]:
        frame = self._frames[self.calls]
        self.calls += 1
        return frame

    def reset(self) -> None:
        self.calls = 0


def _tracker(backend: FakeTrackBackend, **changes: object) -> SubjectInstanceTracker:
    tracking_config = TrackingConfig(**changes)
    instance_config = InstanceSegmentationConfig(
        min_confidence=0.1,
        min_person_height_ratio=0.1,
        sure_fg_erode_px=2,
        subject_unknown_dilate_px=4,
        visitor_bg_dilate_px=2,
        debug_enabled=False,
    )
    return SubjectInstanceTracker(
        instance_config=instance_config,
        tracking_config=tracking_config,
        backend=backend,
    )


def test_tracking_keeps_initial_subject_id_and_marks_other_people_as_visitors(
    tmp_path: Path,
) -> None:
    subject = (18, 20, 72, 98)
    visitor = (74, 20, 96, 70)
    backend = FakeTrackBackend(
        [
            [_candidate(7, subject), _candidate(9, visitor)],
            [_candidate(7, subject), _candidate(9, visitor)],
        ]
    )

    sequence = _tracker(backend).track_paths(_frame_paths(tmp_path, 2))

    assert sequence.status == "ok"
    assert sequence.subject_track_id == 7
    assert [frame.selected.track_id for frame in sequence.frames] == [7, 7]
    assert [[person.track_id for person in frame.visitors] for frame in sequence.frames] == [[9], [9]]
    assert sequence.frames[0].sure_background[40, 80]
    assert sequence.frames[0].sure_foreground[45, 45]


def test_tracking_recovers_subject_for_two_missing_id_frames_by_iou(tmp_path: Path) -> None:
    backend = FakeTrackBackend(
        [
            [_candidate(7, (18, 20, 72, 98))],
            [_candidate(None, (19, 20, 73, 98))],
            [_candidate(None, (20, 20, 74, 98))],
            [_candidate(7, (20, 20, 74, 98))],
        ]
    )

    sequence = _tracker(backend, max_recovery_frames=2, recovery_min_iou=0.35).track_paths(
        _frame_paths(tmp_path, 4)
    )

    assert sequence.status == "ok"
    assert [frame.track_recovered for frame in sequence.frames] == [False, True, True, False]
    assert sequence.track_lost_frames == 2
    assert sequence.track_switch_count == 0


def test_tracking_fails_after_recovery_window_is_exceeded(tmp_path: Path) -> None:
    backend = FakeTrackBackend(
        [
            [_candidate(7, (18, 20, 72, 98))],
            [],
            [],
            [],
        ]
    )

    sequence = _tracker(backend, max_recovery_frames=2).track_paths(_frame_paths(tmp_path, 4))

    assert sequence.status == "tracking_failed"
    assert sequence.subject_track_id == 7
    assert len(sequence.frames) == 4
    assert sequence.track_lost_frames == 3


def test_tracking_fails_when_initial_frame_has_no_primary_subject(tmp_path: Path) -> None:
    backend = FakeTrackBackend([[]])

    sequence = _tracker(backend).track_paths(_frame_paths(tmp_path, 1))

    assert sequence.status == "tracking_failed"
    assert sequence.subject_track_id is None
    assert sequence.error == "no_primary_subject"


def test_tracker_reset_resets_backend_state() -> None:
    backend = FakeTrackBackend([[]])
    backend.calls = 3
    tracker = _tracker(backend)

    tracker.reset()

    assert backend.calls == 0
