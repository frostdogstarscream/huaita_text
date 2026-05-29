from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from PIL import Image, ImageDraw

from subject_alpha_filter import SubjectLocationResult


@dataclass(frozen=True)
class SubjectLocatorConfig:
    enabled: bool = True
    provider: str = "yolo_person_bbox"
    model_path: str = "models/yolo11n.pt"
    min_confidence: float = 0.45
    roi_expand_ratio: float = 0.12
    prefer_center_weight: float = 0.45
    prefer_large_weight: float = 0.35
    prefer_lower_weight: float = 0.20
    min_person_height_ratio: float = 0.25
    roi_side_trim_enabled: bool = True
    roi_side_trim_margin_ratio: float = 0.08
    roi_side_trim_max_overlap_ratio: float = 0.20

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "SubjectLocatorConfig":
        if not isinstance(raw, dict):
            return cls()
        values = {}
        for field_name in cls.__dataclass_fields__:
            if field_name in raw:
                values[field_name] = raw[field_name]
        return cls(**values)


@dataclass(frozen=True)
class SubjectCandidate:
    bbox: tuple[float, float, float, float]
    confidence: float
    score: float | None = None


class SubjectDetector(Protocol):
    def detect(self, source_path: Path) -> list[SubjectCandidate]:
        ...


def _clamp_float(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def choose_primary_subject(
    candidates: list[SubjectCandidate],
    image_size: tuple[int, int],
    config: SubjectLocatorConfig,
) -> SubjectCandidate | None:
    width, height = image_size
    if width <= 0 or height <= 0:
        return None

    scored: list[SubjectCandidate] = []
    min_height = height * float(config.min_person_height_ratio)
    min_conf = float(config.min_confidence)

    for candidate in candidates:
        left, top, right, bottom = candidate.bbox
        box_w = max(float(right) - float(left), 0.0)
        box_h = max(float(bottom) - float(top), 0.0)
        if candidate.confidence < min_conf or box_h < min_height or box_w <= 1:
            continue

        center_x = (float(left) + float(right)) / 2.0
        center_distance = abs(center_x - width / 2.0) / max(width / 2.0, 1.0)
        center_score = 1.0 - _clamp_float(center_distance, 0.0, 1.0)
        area_score = _clamp_float((box_w * box_h) / float(width * height), 0.0, 1.0)
        lower_score = _clamp_float(float(bottom) / float(height), 0.0, 1.0)
        score = (
            float(config.prefer_center_weight) * center_score
            + float(config.prefer_large_weight) * area_score
            + float(config.prefer_lower_weight) * lower_score
        )
        scored.append(replace(candidate, score=score))

    if not scored:
        return None
    return max(scored, key=lambda item: item.score or 0.0)


def expand_bbox(
    bbox: tuple[float, float, float, float],
    image_size: tuple[int, int],
    expand_ratio: float,
) -> tuple[int, int, int, int]:
    width, height = image_size
    left, top, right, bottom = bbox
    box_w = max(float(right) - float(left), 1.0)
    box_h = max(float(bottom) - float(top), 1.0)
    pad_x = box_w * float(expand_ratio)
    pad_y = box_h * float(expand_ratio)

    x1 = int(max(0, round(float(left) - pad_x)))
    y1 = int(max(0, round(float(top) - pad_y)))
    x2 = int(min(width, round(float(right) + pad_x)))
    y2 = int(min(height, round(float(bottom) + pad_y)))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def crop_expanded_roi(
    source_path: Path,
    bbox: tuple[float, float, float, float],
    *,
    output_dir: Path,
    stem: str,
    expand_ratio: float,
) -> Path:
    with Image.open(source_path) as image:
        rgb = image.convert("RGB")
        crop_box = expand_bbox(bbox, rgb.size, expand_ratio)
        roi = rgb.crop(crop_box)

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    roi_path = output_dir / f"{safe_stem}_subject_roi.jpg"
    roi.save(roi_path, format="JPEG", quality=95)
    return roi_path


def crop_roi_box(
    source_path: Path,
    roi_box: tuple[int, int, int, int],
    *,
    output_dir: Path,
    stem: str,
) -> Path:
    with Image.open(source_path) as image:
        rgb = image.convert("RGB")
        roi = rgb.crop(roi_box)

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    roi_path = output_dir / f"{safe_stem}_subject_roi.jpg"
    roi.save(roi_path, format="JPEG", quality=95)
    return roi_path


def _boxes_intersect(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _same_bbox(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    return all(abs(float(x) - float(y)) < 1e-6 for x, y in zip(a, b))


def _intersection_area(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    w = max(min(a[2], b[2]) - max(a[0], b[0]), 0.0)
    h = max(min(a[3], b[3]) - max(a[1], b[1]), 0.0)
    return float(w * h)


def _box_area(box: tuple[float, float, float, float]) -> float:
    return max(float(box[2]) - float(box[0]), 0.0) * max(float(box[3]) - float(box[1]), 0.0)


def trim_roi_away_from_side_visitors(
    roi_box: tuple[int, int, int, int],
    subject: SubjectCandidate,
    visitors: list[SubjectCandidate],
    image_size: tuple[int, int],
    config: SubjectLocatorConfig,
) -> tuple[tuple[int, int, int, int], str]:
    if not config.roi_side_trim_enabled or not visitors:
        return roi_box, ""

    sx1, sy1, sx2, sy2 = subject.bbox
    subject_w = max(float(sx2) - float(sx1), 1.0)
    subject_center_x = (float(sx1) + float(sx2)) / 2.0
    margin = subject_w * float(config.roi_side_trim_margin_ratio)
    max_overlap = float(config.roi_side_trim_max_overlap_ratio)
    left, top, right, bottom = roi_box
    trim_side = ""

    for visitor in visitors:
        if not _boxes_intersect(visitor.bbox, roi_box):
            continue
        visitor_area = max(_box_area(visitor.bbox), 1.0)
        overlap_ratio = _intersection_area(subject.bbox, visitor.bbox) / visitor_area
        if overlap_ratio > max_overlap:
            continue

        vx1, _vy1, vx2, _vy2 = visitor.bbox
        visitor_center_x = (float(vx1) + float(vx2)) / 2.0
        if visitor_center_x > subject_center_x and float(vx1) >= subject_center_x:
            new_right = int(round(float(sx2) + margin))
            if new_right < right:
                right = max(new_right, left + 1)
                trim_side = "right"
        elif visitor_center_x < subject_center_x and float(vx2) <= subject_center_x:
            new_left = int(round(float(sx1) - margin))
            if new_left > left:
                left = min(new_left, right - 1)
                trim_side = "left"

    width, height = image_size
    trimmed = (
        max(0, min(left, width - 1)),
        max(0, min(top, height - 1)),
        max(1, min(right, width)),
        max(1, min(bottom, height)),
    )
    if trimmed[2] <= trimmed[0] or trimmed[3] <= trimmed[1]:
        return roi_box, ""
    return trimmed, trim_side


def save_locator_debug(
    source_path: Path,
    *,
    output_path: Path,
    candidates: list[SubjectCandidate],
    selected: SubjectCandidate,
    roi_box: tuple[int, int, int, int],
) -> None:
    with Image.open(source_path) as image:
        debug = image.convert("RGB")
    draw = ImageDraw.Draw(debug)
    for candidate in candidates:
        color = (0, 220, 0) if _same_bbox(candidate.bbox, selected.bbox) else (255, 80, 80)
        draw.rectangle(candidate.bbox, outline=color, width=3)
    draw.rectangle(roi_box, outline=(60, 160, 255), width=4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    debug.save(output_path, format="JPEG", quality=92)


class YoloPersonBBoxDetector:
    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        self._model = None

    def _load_model(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self._model_path)
        return self._model

    def detect(self, source_path: Path) -> list[SubjectCandidate]:
        model = self._load_model()
        results = model.predict(
            source=str(source_path),
            save=False,
            show=False,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or getattr(boxes, "xyxy", None) is None:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if getattr(boxes, "conf", None) is not None else []
        classes = boxes.cls.cpu().numpy() if getattr(boxes, "cls", None) is not None else []

        candidates: list[SubjectCandidate] = []
        for idx, box in enumerate(xyxy):
            cls_id = int(classes[idx]) if len(classes) > idx else -1
            if cls_id != 0:
                continue
            confidence = float(confs[idx]) if len(confs) > idx else 1.0
            candidates.append(
                SubjectCandidate(
                    bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    confidence=confidence,
                )
            )
        return candidates


class SubjectLocator:
    def __init__(
        self,
        config: SubjectLocatorConfig | dict[str, Any] | None = None,
        *,
        detector: SubjectDetector | None = None,
        output_dir: Path | str | None = None,
    ) -> None:
        self.config = (
            config
            if isinstance(config, SubjectLocatorConfig)
            else SubjectLocatorConfig.from_mapping(config)
        )
        self._detector = detector
        self._output_dir = Path(output_dir) if output_dir is not None else Path("generated") / "subject_rois"
        self._debug_dir = self._output_dir.parent / "subject_debug"
        os.environ.setdefault("YOLO_CONFIG_DIR", str(self._output_dir.parent / "ultralytics"))

    def _get_detector(self) -> SubjectDetector:
        if self._detector is None:
            if self.config.provider != "yolo_person_bbox":
                raise RuntimeError(f"Unsupported subject locator provider: {self.config.provider}")
            self._detector = YoloPersonBBoxDetector(self.config.model_path)
        return self._detector

    def locate(self, source_path: Path, stem: str) -> SubjectLocationResult | None:
        if not self.config.enabled:
            return None
        try:
            with Image.open(source_path) as image:
                image_size = image.size
            detector = self._get_detector()
            candidates = detector.detect(source_path)
            selected = choose_primary_subject(candidates, image_size, self.config)
            if selected is None:
                print(f"[SubjectLocator] no suitable person bbox for {source_path}")
                return None
            roi_box = expand_bbox(selected.bbox, image_size, self.config.roi_expand_ratio)
            candidate_people = [
                candidate
                for candidate in candidates
                if not _same_bbox(candidate.bbox, selected.bbox)
            ]
            roi_box, trim_side = trim_roi_away_from_side_visitors(
                roi_box,
                selected,
                candidate_people,
                image_size,
                self.config,
            )
            roi = crop_roi_box(
                source_path,
                roi_box,
                output_dir=self._output_dir,
                stem=stem,
            )
            other_people = [
                candidate.bbox
                for candidate in candidate_people
                if _boxes_intersect(candidate.bbox, roi_box)
            ]
            max_overlap_ratio = 0.0
            for candidate in candidate_people:
                visitor_area = max(_box_area(candidate.bbox), 1.0)
                max_overlap_ratio = max(
                    max_overlap_ratio,
                    _intersection_area(selected.bbox, candidate.bbox) / visitor_area,
                )
            save_locator_debug(
                source_path,
                output_path=self._debug_dir / f"{stem}_locator.jpg",
                candidates=candidates,
                selected=selected,
                roi_box=roi_box,
            )
            print(
                "[SubjectLocator] selected person "
                f"candidates={len(candidates)} score={selected.score:.3f} "
                f"roi={roi} other_people_in_roi={len(other_people)} "
                f"visitor_overlap_ratio={max_overlap_ratio:.3f} roi_side_trim={trim_side or 'none'}"
            )
            return SubjectLocationResult(
                roi_path=roi,
                original_size=image_size,
                roi_box=roi_box,
                subject=selected,
                candidates=candidates,
                other_person_bboxes=other_people,
                roi_side_trim=trim_side,
                max_visitor_overlap_ratio=max_overlap_ratio,
            )
        except Exception as exc:
            print(f"[SubjectLocator] fallback to original image: {exc}")
            return None
