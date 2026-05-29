from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from PIL import Image

from subject_instance_segmentation import (
    InstanceCandidate,
    InstanceSegmentationConfig,
    build_instance_trimap,
    choose_primary_instance,
)


@dataclass(frozen=True)
class TrackingConfig:
    model_path: str = "models/yolo11x-seg.pt"
    mask_threshold: float = 0.5
    max_recovery_frames: int = 2
    recovery_min_iou: float = 0.35


@dataclass(frozen=True)
class TrackedCandidate:
    track_id: int | None
    bbox: tuple[float, float, float, float]
    confidence: float
    mask: np.ndarray
    score: float | None = None


@dataclass(frozen=True)
class TrackedFrameResult:
    frame_path: Path
    selected: TrackedCandidate | None
    visitors: list[TrackedCandidate]
    trimap: np.ndarray
    sure_foreground: np.ndarray
    sure_background: np.ndarray
    unknown: np.ndarray
    track_recovered: bool = False


@dataclass(frozen=True)
class TrackedInstanceSequence:
    subject_track_id: int | None
    frames: list[TrackedFrameResult]
    track_switch_count: int
    track_lost_frames: int
    status: str
    error: str | None = None


class TrackBackend(Protocol):
    def track(self, frame_path: Path, image_size: tuple[int, int]) -> list[TrackedCandidate]:
        ...


def _as_instance(candidate: TrackedCandidate) -> InstanceCandidate:
    return InstanceCandidate(
        bbox=candidate.bbox,
        confidence=candidate.confidence,
        mask=candidate.mask,
        score=candidate.score,
    )


def _with_score(candidate: TrackedCandidate, selected: InstanceCandidate) -> TrackedCandidate:
    return replace(candidate, score=selected.score)


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.count_nonzero(left.astype(bool) & right.astype(bool)))
    union = int(np.count_nonzero(left.astype(bool) | right.astype(bool)))
    return float(intersection) / float(union) if union else 0.0


def _blank_frame(
    frame_path: Path,
    image_size: tuple[int, int],
    visitors: list[TrackedCandidate],
) -> TrackedFrameResult:
    width, height = image_size
    return TrackedFrameResult(
        frame_path=frame_path,
        selected=None,
        visitors=visitors,
        trimap=np.zeros((height, width), dtype=np.uint8),
        sure_foreground=np.zeros((height, width), dtype=bool),
        sure_background=np.ones((height, width), dtype=bool),
        unknown=np.zeros((height, width), dtype=bool),
    )


class YoloSegTrackBackend:
    def __init__(self, model_path: str, mask_threshold: float = 0.5) -> None:
        self._model_path = model_path
        self._mask_threshold = float(mask_threshold)
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self._model_path)
        return self._model

    def reset(self) -> None:
        self._model = None

    def track(self, frame_path: Path, image_size: tuple[int, int]) -> list[TrackedCandidate]:
        results = self._load_model().track(
            source=str(frame_path),
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            save=False,
            show=False,
            verbose=False,
        )
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)
        if boxes is None or masks is None or boxes.xyxy is None or masks.data is None:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
        ids = boxes.id.cpu().numpy() if boxes.id is not None else [None] * len(xyxy)
        mask_data = masks.data.cpu().numpy()
        width, height = image_size
        candidates: list[TrackedCandidate] = []
        for index, box in enumerate(xyxy):
            if index >= len(mask_data):
                continue
            resized = cv2.resize(mask_data[index].astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
            raw_id = ids[index] if index < len(ids) else None
            track_id = int(raw_id) if raw_id is not None else None
            candidates.append(
                TrackedCandidate(
                    track_id=track_id,
                    bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    confidence=float(confs[index]) if index < len(confs) else 1.0,
                    mask=resized >= self._mask_threshold,
                )
            )
        return candidates


class SubjectInstanceTracker:
    def __init__(
        self,
        instance_config: InstanceSegmentationConfig | None = None,
        tracking_config: TrackingConfig | None = None,
        *,
        backend: TrackBackend | None = None,
    ) -> None:
        self.instance_config = instance_config or InstanceSegmentationConfig()
        self.tracking_config = tracking_config or TrackingConfig(
            model_path=self.instance_config.model_path,
            mask_threshold=self.instance_config.mask_threshold,
        )
        self._backend = backend or YoloSegTrackBackend(
            self.tracking_config.model_path,
            self.tracking_config.mask_threshold,
        )

    def reset(self) -> None:
        reset = getattr(self._backend, "reset", None)
        if callable(reset):
            reset()

    def _frame_result(
        self,
        frame_path: Path,
        image_size: tuple[int, int],
        selected: TrackedCandidate,
        candidates: list[TrackedCandidate],
        recovered: bool,
    ) -> TrackedFrameResult:
        visitors: list[TrackedCandidate] = []
        excluded_selected = False
        for candidate in candidates:
            is_selected_source = (
                not excluded_selected
                and candidate.bbox == selected.bbox
                and np.array_equal(candidate.mask, selected.mask)
            )
            if is_selected_source:
                excluded_selected = True
                continue
            visitors.append(candidate)
        trimap, sure_fg, sure_bg, unknown = build_instance_trimap(
            _as_instance(selected),
            [_as_instance(visitor) for visitor in visitors],
            image_size,
            self.instance_config,
        )
        return TrackedFrameResult(
            frame_path=frame_path,
            selected=selected,
            visitors=visitors,
            trimap=trimap,
            sure_foreground=sure_fg,
            sure_background=sure_bg,
            unknown=unknown,
            track_recovered=recovered,
        )

    def track_paths(self, frame_paths: list[Path]) -> TrackedInstanceSequence:
        if not frame_paths:
            return TrackedInstanceSequence(None, [], 0, 0, "tracking_failed", "no_frames")

        frames: list[TrackedFrameResult] = []
        subject_track_id: int | None = None
        last_selected: TrackedCandidate | None = None
        consecutive_missing = 0
        lost_frames = 0

        for frame_index, frame_path in enumerate(frame_paths):
            with Image.open(frame_path) as image:
                image_size = image.size
            candidates = self._backend.track(frame_path, image_size)

            if frame_index == 0:
                chosen_instance = choose_primary_instance(
                    [_as_instance(candidate) for candidate in candidates],
                    image_size,
                    self.instance_config,
                )
                if chosen_instance is None:
                    return TrackedInstanceSequence(None, [], 0, 0, "tracking_failed", "no_primary_subject")
                chosen_index = next(
                    index
                    for index, candidate in enumerate(candidates)
                    if candidate.bbox == chosen_instance.bbox
                )
                selected = _with_score(candidates[chosen_index], chosen_instance)
                if selected.track_id is None:
                    return TrackedInstanceSequence(None, [], 0, 0, "tracking_failed", "no_primary_track_id")
                subject_track_id = selected.track_id
                last_selected = selected
                frames.append(self._frame_result(frame_path, image_size, selected, candidates, False))
                continue

            matching = next(
                (candidate for candidate in candidates if candidate.track_id == subject_track_id),
                None,
            )
            if matching is not None:
                consecutive_missing = 0
                last_selected = matching
                frames.append(self._frame_result(frame_path, image_size, matching, candidates, False))
                continue

            consecutive_missing += 1
            lost_frames += 1
            if consecutive_missing > self.tracking_config.max_recovery_frames:
                frames.append(_blank_frame(frame_path, image_size, candidates))
                return TrackedInstanceSequence(
                    subject_track_id,
                    frames,
                    0,
                    lost_frames,
                    "tracking_failed",
                    "subject_track_lost",
                )

            recovery = None
            if last_selected is not None and candidates:
                best = max(candidates, key=lambda candidate: _mask_iou(last_selected.mask, candidate.mask))
                if _mask_iou(last_selected.mask, best.mask) >= self.tracking_config.recovery_min_iou:
                    recovery = replace(best, track_id=subject_track_id)
            if recovery is None:
                frames.append(_blank_frame(frame_path, image_size, candidates))
                continue
            last_selected = recovery
            frames.append(self._frame_result(frame_path, image_size, recovery, candidates, True))

        return TrackedInstanceSequence(subject_track_id, frames, 0, lost_frames, "ok")
