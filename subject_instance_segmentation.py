from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class InstanceSegmentationConfig:
    enabled: bool = True
    provider: str = "yolo_person_seg"
    model_path: str = "models/yolo11x-seg.pt"
    min_confidence: float = 0.45
    min_person_height_ratio: float = 0.25
    prefer_center_weight: float = 0.45
    prefer_large_weight: float = 0.35
    prefer_lower_weight: float = 0.20
    mask_threshold: float = 0.5
    sure_fg_erode_px: int = 10
    subject_unknown_dilate_px: int = 18
    visitor_bg_dilate_px: int = 18
    debug_enabled: bool = True

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "InstanceSegmentationConfig":
        if not isinstance(raw, dict):
            return cls()
        values = {}
        for field_name in cls.__dataclass_fields__:
            if field_name in raw:
                values[field_name] = raw[field_name]
        return cls(**values)


@dataclass(frozen=True)
class InstanceCandidate:
    bbox: tuple[float, float, float, float]
    confidence: float
    mask: np.ndarray
    score: float | None = None


@dataclass(frozen=True)
class InstanceSegmentationResult:
    source_path: Path
    image_size: tuple[int, int]
    selected: InstanceCandidate
    candidates: list[InstanceCandidate]
    visitors: list[InstanceCandidate]
    trimap: np.ndarray
    sure_foreground: np.ndarray
    sure_background: np.ndarray
    unknown: np.ndarray


class InstanceSegmentDetector(Protocol):
    def detect(self, source_path: Path, image_size: tuple[int, int]) -> list[InstanceCandidate]:
        ...


def _clamp_float(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def _same_bbox(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return all(abs(float(x) - float(y)) < 1e-6 for x, y in zip(a, b))


def _mask_area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def choose_primary_instance(
    candidates: list[InstanceCandidate],
    image_size: tuple[int, int],
    config: InstanceSegmentationConfig,
) -> InstanceCandidate | None:
    width, height = image_size
    if width <= 0 or height <= 0:
        return None

    scored: list[InstanceCandidate] = []
    min_height = height * float(config.min_person_height_ratio)
    for candidate in candidates:
        left, top, right, bottom = candidate.bbox
        box_w = max(float(right) - float(left), 0.0)
        box_h = max(float(bottom) - float(top), 0.0)
        if candidate.confidence < float(config.min_confidence) or box_h < min_height or box_w <= 1:
            continue
        if _mask_area(candidate.mask) <= 0:
            continue

        center_x = (float(left) + float(right)) / 2.0
        center_distance = abs(center_x - width / 2.0) / max(width / 2.0, 1.0)
        center_score = 1.0 - _clamp_float(center_distance, 0.0, 1.0)
        area_score = _clamp_float(_mask_area(candidate.mask) / float(width * height), 0.0, 1.0)
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


def _morph(mask: np.ndarray, operation: int, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    size = int(radius) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.morphologyEx(mask.astype(np.uint8), operation, kernel).astype(bool)


def build_instance_trimap(
    selected: InstanceCandidate,
    visitors: list[InstanceCandidate],
    image_size: tuple[int, int],
    config: InstanceSegmentationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    width, height = image_size
    subject_mask = selected.mask.astype(bool)
    sure_fg = _morph(subject_mask, cv2.MORPH_ERODE, int(config.sure_fg_erode_px))
    subject_dilated = _morph(subject_mask, cv2.MORPH_DILATE, int(config.subject_unknown_dilate_px))
    unknown = subject_dilated & ~sure_fg

    sure_bg = ~subject_dilated
    for visitor in visitors:
        visitor_mask = _morph(visitor.mask.astype(bool), cv2.MORPH_DILATE, int(config.visitor_bg_dilate_px))
        sure_bg |= visitor_mask

    sure_fg &= ~sure_bg
    unknown &= ~sure_bg
    trimap = np.zeros((height, width), dtype=np.uint8)
    trimap[unknown] = 128
    trimap[sure_fg] = 255
    return trimap, sure_fg, sure_bg, unknown


def cutout_from_instance_mask(source_path: Path, selected: InstanceCandidate) -> Image.Image:
    with Image.open(source_path) as image:
        rgba = image.convert("RGBA")
    arr = np.array(rgba)
    alpha = np.where(selected.mask.astype(bool), 255, 0).astype(np.uint8)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=1.0, sigmaY=1.0)
    arr[:, :, 3] = np.minimum(arr[:, :, 3], alpha)
    return Image.fromarray(arr, "RGBA")


class YoloPersonSegDetector:
    def __init__(self, model_path: str, mask_threshold: float = 0.5) -> None:
        self._model_path = model_path
        self._mask_threshold = float(mask_threshold)
        self._model = None

    def _load_model(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self._model_path)
        return self._model

    def detect(self, source_path: Path, image_size: tuple[int, int]) -> list[InstanceCandidate]:
        model = self._load_model()
        results = model.predict(source=str(source_path), save=False, show=False, verbose=False)
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)
        if boxes is None or masks is None or getattr(boxes, "xyxy", None) is None or getattr(masks, "data", None) is None:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if getattr(boxes, "conf", None) is not None else []
        classes = boxes.cls.cpu().numpy() if getattr(boxes, "cls", None) is not None else []
        mask_data = masks.data.cpu().numpy()
        width, height = image_size

        candidates: list[InstanceCandidate] = []
        for idx, box in enumerate(xyxy):
            cls_id = int(classes[idx]) if len(classes) > idx else -1
            if cls_id != 0 or idx >= len(mask_data):
                continue
            raw_mask = mask_data[idx]
            resized = cv2.resize(raw_mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
            mask = resized >= self._mask_threshold
            confidence = float(confs[idx]) if len(confs) > idx else 1.0
            candidates.append(
                InstanceCandidate(
                    bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    confidence=confidence,
                    mask=mask,
                )
            )
        return candidates


class SubjectInstanceSegmenter:
    def __init__(
        self,
        config: InstanceSegmentationConfig | dict[str, Any] | None = None,
        *,
        detector: InstanceSegmentDetector | None = None,
        output_dir: Path | str | None = None,
    ) -> None:
        self.config = config if isinstance(config, InstanceSegmentationConfig) else InstanceSegmentationConfig.from_mapping(config)
        self._detector = detector
        self._output_dir = Path(output_dir) if output_dir is not None else Path("generated") / "instance_seg"
        os.environ.setdefault("YOLO_CONFIG_DIR", str(self._output_dir.parent / "ultralytics"))

    def _get_detector(self) -> InstanceSegmentDetector:
        if self._detector is None:
            if self.config.provider != "yolo_person_seg":
                raise RuntimeError(f"Unsupported instance segmentation provider: {self.config.provider}")
            self._detector = YoloPersonSegDetector(self.config.model_path, self.config.mask_threshold)
        return self._detector

    def segment(self, source_path: Path, stem: str) -> InstanceSegmentationResult | None:
        if not self.config.enabled:
            return None
        try:
            with Image.open(source_path) as image:
                image_size = image.size
            candidates = self._get_detector().detect(source_path, image_size)
            selected = choose_primary_instance(candidates, image_size, self.config)
            if selected is None:
                print(f"[SubjectInstanceSeg] no suitable person instance for {source_path}")
                return None
            visitors = [candidate for candidate in candidates if not _same_bbox(candidate.bbox, selected.bbox)]
            trimap, sure_fg, sure_bg, unknown = build_instance_trimap(selected, visitors, image_size, self.config)
            result = InstanceSegmentationResult(
                source_path=source_path,
                image_size=image_size,
                selected=selected,
                candidates=candidates,
                visitors=visitors,
                trimap=trimap,
                sure_foreground=sure_fg,
                sure_background=sure_bg,
                unknown=unknown,
            )
            if self.config.debug_enabled:
                save_instance_debug(source_path, result, self._output_dir, stem)
            print(
                "[SubjectInstanceSeg] "
                f"candidates={len(candidates)} visitors={len(visitors)} "
                f"selected_score={selected.score:.3f} selected_mask_px={_mask_area(selected.mask)}"
            )
            return result
        except Exception as exc:
            print(f"[SubjectInstanceSeg] failed: {exc}")
            return None


def _mask_overlay(size: tuple[int, int], mask: np.ndarray, color: tuple[int, int, int, int]) -> Image.Image:
    arr = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    arr[mask.astype(bool)] = color
    return Image.fromarray(arr, "RGBA")


def save_instance_debug(
    source_path: Path,
    result: InstanceSegmentationResult,
    output_dir: Path,
    stem: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        base = image.convert("RGBA")

    overlay = base.copy()
    overlay.alpha_composite(_mask_overlay(base.size, result.selected.mask, (0, 255, 0, 95)))
    for visitor in result.visitors:
        overlay.alpha_composite(_mask_overlay(base.size, visitor.mask, (255, 0, 0, 95)))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(result.selected.bbox, outline=(0, 255, 0, 255), width=3)
    for visitor in result.visitors:
        draw.rectangle(visitor.bbox, outline=(255, 0, 0, 255), width=2)
    overlay.convert("RGB").save(output_dir / f"{stem}_seg_instance_debug.jpg", format="JPEG", quality=92)

    Image.fromarray((result.selected.mask.astype(np.uint8) * 255), "L").save(output_dir / f"{stem}_subject_mask.png", format="PNG")
    visitor_mask = np.zeros((base.height, base.width), dtype=bool)
    for visitor in result.visitors:
        visitor_mask |= visitor.mask.astype(bool)
    Image.fromarray((visitor_mask.astype(np.uint8) * 255), "L").save(output_dir / f"{stem}_visitor_mask.png", format="PNG")
    Image.fromarray(result.trimap, "L").save(output_dir / f"{stem}_trimap.png", format="PNG")
