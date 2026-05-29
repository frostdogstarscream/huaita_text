from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from subject_alpha_filter import SubjectLocationResult, map_bbox_to_output


@dataclass(frozen=True)
class SubjectVisitorSuppressionConfig:
    enabled: bool = True
    pre_aliyun_enabled: bool = True
    post_alpha_hard_clear: bool = True
    visitor_preclean_expand_ratio: float = 0.18
    subject_protect_expand_ratio: float = 0.04
    fill_mode: str = "inpaint"
    inpaint_radius: int = 9
    debug_enabled: bool = True

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "SubjectVisitorSuppressionConfig":
        if not isinstance(raw, dict):
            return cls()
        values = {}
        for field_name in cls.__dataclass_fields__:
            if field_name in raw:
                values[field_name] = raw[field_name]
        return cls(**values)


@dataclass(frozen=True)
class VisitorSuppressionResult:
    cleaned_roi_path: Path
    visitor_mask_pixels: int
    visitor_boxes: list[tuple[int, int, int, int]]
    protect_box: tuple[int, int, int, int] | None


@dataclass(frozen=True)
class VisitorMaskSuppressionResult:
    image: Image.Image
    visitor_mask_pixels: int


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
    box_w = max(float(right) - float(left), 1.0)
    box_h = max(float(bottom) - float(top), 1.0)
    pad_x = box_w * float(ratio)
    pad_y = box_h * float(ratio)
    return _clamp_box((left - pad_x, top - pad_y, right + pad_x, bottom + pad_y), size)


def _mask_from_boxes(size: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    mask = np.zeros((size[1], size[0]), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        mask[y1:y2, x1:x2] = True
    return mask


def build_visitor_suppression_mask(
    location: SubjectLocationResult,
    output_size: tuple[int, int],
    config: SubjectVisitorSuppressionConfig,
    *,
    visitor_expand_ratio: float | None = None,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]], tuple[int, int, int, int] | None]:
    if not location.other_person_bboxes:
        return np.zeros((output_size[1], output_size[0]), dtype=bool), [], None

    expand_ratio = (
        config.visitor_preclean_expand_ratio
        if visitor_expand_ratio is None
        else visitor_expand_ratio
    )
    visitor_boxes = [
        _expand_box(
            map_bbox_to_output(box, roi_box=location.roi_box, output_size=output_size),
            output_size,
            expand_ratio,
        )
        for box in location.other_person_bboxes
    ]
    subject_box = map_bbox_to_output(location.subject.bbox, roi_box=location.roi_box, output_size=output_size)
    protect_box = _expand_box(subject_box, output_size, config.subject_protect_expand_ratio)
    visitor_mask = _mask_from_boxes(output_size, visitor_boxes)
    protect_mask = _mask_from_boxes(output_size, [protect_box])
    return visitor_mask & ~protect_mask, visitor_boxes, protect_box


def suppress_visitors_in_roi(
    roi_path: Path,
    location: SubjectLocationResult,
    *,
    output_dir: Path,
    stem: str,
    config: SubjectVisitorSuppressionConfig,
) -> VisitorSuppressionResult:
    if not config.enabled or not config.pre_aliyun_enabled:
        return VisitorSuppressionResult(roi_path, 0, [], None)

    with Image.open(roi_path) as image:
        rgb = image.convert("RGB")

    visitor_mask, visitor_boxes, protect_box = build_visitor_suppression_mask(
        location,
        rgb.size,
        config,
    )
    mask_pixels = int(np.count_nonzero(visitor_mask))
    if mask_pixels <= 0:
        return VisitorSuppressionResult(roi_path, 0, visitor_boxes, protect_box)

    arr = np.array(rgb)
    if config.fill_mode == "inpaint":
        inpaint_mask = (visitor_mask * 255).astype(np.uint8)
        arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        filled_bgr = cv2.inpaint(arr_bgr, inpaint_mask, config.inpaint_radius, cv2.INPAINT_TELEA)
        arr = cv2.cvtColor(filled_bgr, cv2.COLOR_BGR2RGB)
    elif config.fill_mode == "solid_background":
        background = np.zeros_like(arr)
        border_pixels = np.concatenate(
            [arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :]],
            axis=0,
        )
        color = np.median(border_pixels, axis=0).astype(np.uint8)
        background[:, :] = color
        arr[visitor_mask] = background[visitor_mask]
    else:
        background = np.array(rgb.filter(ImageFilter.GaussianBlur(radius=18)))
        arr[visitor_mask] = background[visitor_mask]

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    cleaned_path = output_dir / f"{safe_stem}_subject_cleaned_roi.jpg"
    Image.fromarray(arr, "RGB").save(cleaned_path, format="JPEG", quality=95)

    if config.debug_enabled:
        save_visitor_mask_debug(
            rgb.size,
            visitor_mask,
            visitor_boxes,
            protect_box,
            output_dir.parent / "subject_debug" / f"{safe_stem}_visitor_mask.png",
        )

    return VisitorSuppressionResult(cleaned_path, mask_pixels, visitor_boxes, protect_box)


def apply_post_alpha_hard_clear(
    image: Image.Image,
    location: SubjectLocationResult,
    config: SubjectVisitorSuppressionConfig,
) -> Image.Image:
    if not config.enabled or not config.post_alpha_hard_clear:
        return image
    rgba = image.convert("RGBA")
    visitor_mask, _visitor_boxes, _protect_box = build_visitor_suppression_mask(
        location,
        rgba.size,
        config,
        visitor_expand_ratio=config.visitor_preclean_expand_ratio,
    )
    if not np.any(visitor_mask):
        return rgba
    arr = np.array(rgba)
    arr[:, :, 3] = np.where(visitor_mask, 0, arr[:, :, 3]).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def _dilate_bool_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return mask.astype(bool)
    size = radius_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def build_visitor_suppression_mask_from_instance(
    instance_result: Any,
    output_size: tuple[int, int],
    config: SubjectVisitorSuppressionConfig,
) -> np.ndarray:
    width, height = output_size
    visitor_mask = np.zeros((height, width), dtype=bool)
    for visitor in getattr(instance_result, "visitors", []):
        visitor_mask |= np.asarray(visitor.mask).astype(bool)
    if not np.any(visitor_mask):
        return visitor_mask
    subject_mask = np.asarray(instance_result.selected.mask).astype(bool)
    min_side = max(min(width, height), 1)
    dilate_px = max(int(round(float(config.visitor_preclean_expand_ratio) * min_side)), 0)
    expanded_visitor = _dilate_bool_mask(visitor_mask, dilate_px)
    protect_px = max(int(round(float(config.subject_protect_expand_ratio) * min_side)), 0)
    expanded_subject = _dilate_bool_mask(subject_mask, protect_px)
    return expanded_visitor & ~expanded_subject


def suppress_visitors_with_instance_masks(
    image: Image.Image,
    instance_result: Any,
    config: SubjectVisitorSuppressionConfig,
) -> VisitorMaskSuppressionResult:
    if not config.enabled or not config.pre_aliyun_enabled:
        return VisitorMaskSuppressionResult(image=image.convert("RGB"), visitor_mask_pixels=0)
    rgb = image.convert("RGB")
    visitor_mask = build_visitor_suppression_mask_from_instance(instance_result, rgb.size, config)
    mask_pixels = int(np.count_nonzero(visitor_mask))
    if mask_pixels <= 0:
        return VisitorMaskSuppressionResult(image=rgb, visitor_mask_pixels=0)

    arr = np.array(rgb)
    inpaint_mask = (visitor_mask * 255).astype(np.uint8)
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    filled_bgr = cv2.inpaint(arr_bgr, inpaint_mask, config.inpaint_radius, cv2.INPAINT_TELEA)
    cleaned = Image.fromarray(cv2.cvtColor(filled_bgr, cv2.COLOR_BGR2RGB), "RGB")
    return VisitorMaskSuppressionResult(image=cleaned, visitor_mask_pixels=mask_pixels)


def apply_post_alpha_hard_clear_with_instance(
    image: Image.Image,
    instance_result: Any,
    config: SubjectVisitorSuppressionConfig,
) -> Image.Image:
    if not config.enabled or not config.post_alpha_hard_clear:
        return image.convert("RGBA")
    rgba = image.convert("RGBA")
    visitor_mask = build_visitor_suppression_mask_from_instance(instance_result, rgba.size, config)
    if not np.any(visitor_mask):
        return rgba
    arr = np.array(rgba)
    arr[:, :, 3] = np.where(visitor_mask, 0, arr[:, :, 3]).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def save_visitor_mask_debug(
    size: tuple[int, int],
    visitor_mask: np.ndarray,
    visitor_boxes: list[tuple[int, int, int, int]],
    protect_box: tuple[int, int, int, int] | None,
    output_path: Path,
) -> None:
    debug = Image.new("RGBA", size, (0, 0, 0, 0))
    overlay = np.array(debug)
    overlay[visitor_mask] = (255, 0, 0, 110)
    debug = Image.fromarray(overlay, "RGBA")
    draw = ImageDraw.Draw(debug)
    for box in visitor_boxes:
        draw.rectangle(box, outline=(255, 0, 0, 255), width=2)
    if protect_box is not None:
        draw.rectangle(protect_box, outline=(0, 255, 0, 255), width=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    debug.save(output_path, format="PNG")
