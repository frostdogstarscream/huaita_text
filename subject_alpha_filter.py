from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class SubjectAlphaFilterConfig:
    enabled: bool = True
    mode: str = "strong_remove_visitors"
    alpha_threshold: int = 8
    subject_box_expand_ratio: float = 0.08
    visitor_box_expand_ratio: float = 0.18
    keep_nearby_component_px: int = 6
    morph_close_kernel_px: int = 20
    debug_enabled: bool = True

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "SubjectAlphaFilterConfig":
        if not isinstance(raw, dict):
            return cls()
        values = {}
        for field_name in cls.__dataclass_fields__:
            if field_name in raw:
                values[field_name] = raw[field_name]
        return cls(**values)


@dataclass(frozen=True)
class SubjectLocationResult:
    roi_path: Path
    original_size: tuple[int, int]
    roi_box: tuple[int, int, int, int]
    subject: Any
    candidates: list[Any]
    other_person_bboxes: list[tuple[float, float, float, float]]
    roi_side_trim: str = ""
    max_visitor_overlap_ratio: float = 0.0


def _clamp_box(box: tuple[float, float, float, float], size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = size
    left, top, right, bottom = box
    x1 = max(0, min(int(round(left)), width))
    y1 = max(0, min(int(round(top)), height))
    x2 = max(0, min(int(round(right)), width))
    y2 = max(0, min(int(round(bottom)), height))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def _expand_box(
    box: tuple[float, float, float, float],
    size: tuple[int, int],
    ratio: float,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    width = max(float(right) - float(left), 1.0)
    height = max(float(bottom) - float(top), 1.0)
    pad_x = width * float(ratio)
    pad_y = height * float(ratio)
    return _clamp_box((left - pad_x, top - pad_y, right + pad_x, bottom + pad_y), size)


def map_bbox_to_output(
    bbox: tuple[float, float, float, float],
    *,
    roi_box: tuple[int, int, int, int],
    output_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    roi_left, roi_top, roi_right, roi_bottom = roi_box
    roi_w = max(roi_right - roi_left, 1)
    roi_h = max(roi_bottom - roi_top, 1)
    out_w, out_h = output_size
    scale_x = out_w / float(roi_w)
    scale_y = out_h / float(roi_h)
    left, top, right, bottom = bbox
    mapped = (
        (float(left) - roi_left) * scale_x,
        (float(top) - roi_top) * scale_y,
        (float(right) - roi_left) * scale_x,
        (float(bottom) - roi_top) * scale_y,
    )
    return _clamp_box(mapped, output_size)


def _boxes_intersect(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _box_mask(size: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    mask = np.zeros((size[1], size[0]), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        mask[y1:y2, x1:x2] = True
    return mask


def _component_touches_subject(
    label_mask: np.ndarray,
    subject_mask: np.ndarray,
    keep_nearby_px: int,
) -> bool:
    if np.any(label_mask & subject_mask):
        return True
    if keep_nearby_px <= 0:
        return False
    kernel_size = keep_nearby_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    nearby = cv2.dilate(subject_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return bool(np.any(label_mask & nearby))


def filter_subject_alpha(
    image: Image.Image,
    location: SubjectLocationResult,
    config: SubjectAlphaFilterConfig,
) -> Image.Image:
    if not config.enabled:
        return image

    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    alpha = arr[:, :, 3]
    output_size = rgba.size
    foreground = alpha > int(config.alpha_threshold)
    if not np.any(foreground):
        return rgba

    subject_box = map_bbox_to_output(location.subject.bbox, roi_box=location.roi_box, output_size=output_size)
    subject_keep_box = _expand_box(subject_box, output_size, config.subject_box_expand_ratio)
    subject_mask = _box_mask(output_size, [subject_keep_box])

    visitor_boxes = [
        _expand_box(
            map_bbox_to_output(box, roi_box=location.roi_box, output_size=output_size),
            output_size,
            config.visitor_box_expand_ratio,
        )
        for box in location.other_person_bboxes
        if _boxes_intersect(box, location.roi_box)
    ]
    visitor_mask = _box_mask(output_size, visitor_boxes) & ~subject_mask

    analysis_mask = foreground & ~visitor_mask
    if not np.any(analysis_mask):
        return rgba

    cc_input = analysis_mask.astype(np.uint8)
    if config.morph_close_kernel_px > 0:
        k = config.morph_close_kernel_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        cc_input = cv2.morphologyEx(cc_input, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        cc_input,
        connectivity=8,
    )
    keep = np.zeros_like(foreground, dtype=bool)
    best_label = 0
    best_overlap = 0
    best_area = 0
    for label in range(1, count):
        label_mask = labels == label
        overlap = int(np.count_nonzero(label_mask & subject_mask))
        area = int(stats[label, cv2.CC_STAT_AREA])
        if overlap > best_overlap or (overlap == best_overlap and area > best_area):
            best_label = label
            best_overlap = overlap
            best_area = area

    if best_label == 0 or best_overlap <= 0:
        return rgba

    for label in range(1, count):
        label_mask = labels == label
        if label == best_label or _component_touches_subject(
            label_mask,
            labels == best_label,
            int(config.keep_nearby_component_px),
        ):
            keep |= label_mask

    new_alpha = np.where(keep, alpha, 0).astype(np.uint8)
    arr[:, :, 3] = new_alpha
    return Image.fromarray(arr, "RGBA")


def save_alpha_filter_debug(
    image: Image.Image,
    location: SubjectLocationResult,
    output_path: Path,
    config: SubjectAlphaFilterConfig,
) -> None:
    if not config.debug_enabled:
        return
    debug = image.convert("RGBA")
    draw = ImageDraw.Draw(debug)
    subject_box = map_bbox_to_output(location.subject.bbox, roi_box=location.roi_box, output_size=debug.size)
    draw.rectangle(subject_box, outline=(0, 255, 0, 255), width=3)
    for box in location.other_person_bboxes:
        mapped = map_bbox_to_output(box, roi_box=location.roi_box, output_size=debug.size)
        draw.rectangle(mapped, outline=(255, 0, 0, 255), width=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    debug.save(output_path, format="PNG")


def _dilate_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return mask.astype(bool)
    size = radius_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def filter_subject_alpha_with_instance(
    image: Image.Image,
    instance_result: Any,
    config: SubjectAlphaFilterConfig,
) -> Image.Image:
    if not config.enabled:
        return image.convert("RGBA")

    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    alpha = arr[:, :, 3]
    foreground = alpha > int(config.alpha_threshold)
    if not np.any(foreground):
        return rgba

    width, height = rgba.size
    min_side = max(min(width, height), 1)
    subject_expand_px = max(int(round(float(config.subject_box_expand_ratio) * min_side)), 0)
    visitor_expand_px = max(int(round(float(config.visitor_box_expand_ratio) * min_side)), 0)

    subject_mask = _dilate_mask(np.asarray(instance_result.selected.mask).astype(bool), subject_expand_px)
    visitor_mask = np.zeros_like(subject_mask, dtype=bool)
    for visitor in getattr(instance_result, "visitors", []):
        visitor_mask |= _dilate_mask(np.asarray(visitor.mask).astype(bool), visitor_expand_px)
    visitor_mask &= ~subject_mask

    analysis_mask = foreground & ~visitor_mask
    if not np.any(analysis_mask):
        return rgba

    cc_input = analysis_mask.astype(np.uint8)
    if config.morph_close_kernel_px > 0:
        k = config.morph_close_kernel_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        cc_input = cv2.morphologyEx(cc_input, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(cc_input, connectivity=8)
    keep = np.zeros_like(foreground, dtype=bool)
    best_label = 0
    best_overlap = 0
    best_area = 0
    for label in range(1, count):
        label_mask = labels == label
        overlap = int(np.count_nonzero(label_mask & subject_mask))
        area = int(stats[label, cv2.CC_STAT_AREA])
        if overlap > best_overlap or (overlap == best_overlap and area > best_area):
            best_label = label
            best_overlap = overlap
            best_area = area

    if best_label == 0 or best_overlap <= 0:
        return rgba

    subject_component = labels == best_label
    for label in range(1, count):
        label_mask = labels == label
        if label == best_label or _component_touches_subject(
            label_mask,
            subject_component,
            int(config.keep_nearby_component_px),
        ):
            keep |= label_mask

    arr[:, :, 3] = np.where(keep, alpha, 0).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")
