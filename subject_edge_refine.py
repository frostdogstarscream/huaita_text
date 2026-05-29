from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SubjectEdgeRefineConfig:
    enabled: bool = True
    min_component_area_ratio: float = 0.0012
    open_kernel_px: int = 1
    feather_radius_px: float = 1.3
    edge_ring_blur_enabled: bool = True
    edge_ring_inner_px: int = 3
    edge_ring_outer_px: int = 3
    edge_ring_sigma: float = 1.0
    arm_edge_tighten_enabled: bool = True
    arm_edge_tighten_px: int = 1
    arm_edge_tighten_strength: str = "medium"
    hard_clear_feather_px: int = 4
    effective_bbox_alpha_threshold: int = 16
    debug_enabled: bool = True

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "SubjectEdgeRefineConfig":
        if not isinstance(raw, dict):
            return cls()
        values = {}
        for field_name in cls.__dataclass_fields__:
            if field_name in raw:
                values[field_name] = raw[field_name]
        return cls(**values)


@dataclass(frozen=True)
class SubjectEdgeRefineResult:
    image: Image.Image
    removed_small_components: int
    alpha_area_before: int
    alpha_area_after: int
    effective_bbox: tuple[int, int, int, int] | None
    edge_ring_blurred_px: int
    edge_ring_alpha_delta_mean: float
    edge_ring_mask: np.ndarray | None
    arm_edge_tighten_applied_px: int
    arm_edge_tighten_mask: np.ndarray | None


def _morph(mask: np.ndarray, operation: int, radius: int) -> np.ndarray:
    if int(radius) <= 0:
        return mask.astype(bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(radius) * 2 + 1, int(radius) * 2 + 1))
    return cv2.morphologyEx(mask.astype(np.uint8), operation, kernel).astype(bool)


def _build_edge_ring_mask(subject_mask: np.ndarray, *, inner_px: int, outer_px: int) -> np.ndarray:
    inner_boundary = subject_mask & ~_morph(subject_mask, cv2.MORPH_ERODE, int(inner_px))
    outer_boundary = _morph(subject_mask, cv2.MORPH_DILATE, int(outer_px)) & ~subject_mask
    return inner_boundary | outer_boundary


def _build_arm_tighten_mask(subject_mask: np.ndarray) -> np.ndarray:
    if not np.any(subject_mask):
        return np.zeros_like(subject_mask, dtype=bool)
    ys, xs = np.where(subject_mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    w = max(x1 - x0, 1)
    h = max(y1 - y0, 1)

    head_cutoff = y0 + int(0.42 * h)
    side_band = max(int(0.34 * w), 1)
    left_band = np.zeros_like(subject_mask, dtype=bool)
    right_band = np.zeros_like(subject_mask, dtype=bool)
    left_band[:, x0 : min(x0 + side_band, subject_mask.shape[1])] = True
    right_band[:, max(x1 - side_band, 0) : x1] = True

    lower = np.zeros_like(subject_mask, dtype=bool)
    lower[head_cutoff:y1, :] = True
    arm_zone = subject_mask & lower & (left_band | right_band)

    inner_boundary = subject_mask & ~_morph(subject_mask, cv2.MORPH_ERODE, 1)
    return arm_zone & inner_boundary


def effective_alpha_bbox(
    image: Image.Image,
    *,
    alpha_threshold: int = 16,
) -> tuple[int, int, int, int] | None:
    rgba = image.convert("RGBA")
    alpha = np.array(rgba.getchannel("A"))
    mask = alpha > int(alpha_threshold)
    if not np.any(mask):
        return None
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _remove_small_components(
    mask: np.ndarray,
    *,
    min_area: int,
) -> tuple[np.ndarray, int]:
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return mask, 0

    largest_label = 1
    largest_area = int(stats[1, cv2.CC_STAT_AREA])
    for label in range(2, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > largest_area:
            largest_label = label
            largest_area = area

    keep = labels == largest_label
    removed = 0
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if label == largest_label or area >= min_area:
            keep |= labels == label
        else:
            removed += 1
    return keep, removed


def refine_subject_edge(
    image: Image.Image,
    config: SubjectEdgeRefineConfig,
) -> SubjectEdgeRefineResult:
    rgba = image.convert("RGBA")
    if not config.enabled:
        return SubjectEdgeRefineResult(
            rgba,
            0,
            int(np.count_nonzero(np.array(rgba.getchannel("A")) > 0)),
            int(np.count_nonzero(np.array(rgba.getchannel("A")) > 0)),
            effective_alpha_bbox(rgba, alpha_threshold=config.effective_bbox_alpha_threshold),
            0,
            0.0,
            None,
            0,
            None,
        )

    arr = np.array(rgba)
    alpha = arr[:, :, 3]
    foreground = alpha > 0
    alpha_area_before = int(np.count_nonzero(foreground))
    if alpha_area_before <= 0:
        return SubjectEdgeRefineResult(rgba, 0, 0, 0, None, 0, 0.0, None, 0, None)

    min_area = max(int(alpha_area_before * float(config.min_component_area_ratio)), 16)
    keep_mask, removed_small = _remove_small_components(foreground, min_area=min_area)

    if config.open_kernel_px > 0:
        k = int(config.open_kernel_px) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        keep_mask = cv2.morphologyEx(keep_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)

    arm_edge_tighten_mask: np.ndarray | None = None
    arm_edge_tighten_applied_px = 0
    if config.arm_edge_tighten_enabled and int(config.arm_edge_tighten_px) > 0:
        arm_edge_tighten_mask = _build_arm_tighten_mask(keep_mask)
        if np.any(arm_edge_tighten_mask):
            eroded_subject = _morph(keep_mask, cv2.MORPH_ERODE, int(config.arm_edge_tighten_px))
            keep_mask = np.where(arm_edge_tighten_mask, eroded_subject, keep_mask).astype(bool)
            arm_edge_tighten_applied_px = int(np.count_nonzero(arm_edge_tighten_mask))

    refined_alpha = np.where(keep_mask, alpha, 0).astype(np.uint8)
    if config.feather_radius_px > 0:
        refined_alpha = cv2.GaussianBlur(
            refined_alpha,
            (0, 0),
            sigmaX=float(config.feather_radius_px),
            sigmaY=float(config.feather_radius_px),
        )
        refined_alpha = np.where(keep_mask | (refined_alpha > 0), refined_alpha, 0).astype(np.uint8)

    edge_ring_mask: np.ndarray | None = None
    edge_ring_blurred_px = 0
    edge_ring_alpha_delta_mean = 0.0
    if config.edge_ring_blur_enabled and config.edge_ring_sigma > 0:
        edge_ring_mask = _build_edge_ring_mask(
            keep_mask,
            inner_px=int(config.edge_ring_inner_px),
            outer_px=int(config.edge_ring_outer_px),
        )
        if np.any(edge_ring_mask):
            ring_blurred = cv2.GaussianBlur(
                refined_alpha,
                (0, 0),
                sigmaX=float(config.edge_ring_sigma),
                sigmaY=float(config.edge_ring_sigma),
            )
            alpha_before_ring = refined_alpha.copy()
            refined_alpha = np.where(edge_ring_mask, ring_blurred, refined_alpha).astype(np.uint8)
            delta = np.abs(refined_alpha.astype(np.int16) - alpha_before_ring.astype(np.int16))
            edge_ring_blurred_px = int(np.count_nonzero(edge_ring_mask))
            edge_ring_alpha_delta_mean = float(np.mean(delta[edge_ring_mask])) if edge_ring_blurred_px else 0.0

    arr[:, :, 3] = refined_alpha
    refined = Image.fromarray(arr, "RGBA")
    alpha_area_after = int(np.count_nonzero(refined_alpha > 0))
    return SubjectEdgeRefineResult(
        refined,
        removed_small,
        alpha_area_before,
        alpha_area_after,
        effective_alpha_bbox(refined, alpha_threshold=config.effective_bbox_alpha_threshold),
        edge_ring_blurred_px,
        edge_ring_alpha_delta_mean,
        edge_ring_mask,
        arm_edge_tighten_applied_px,
        arm_edge_tighten_mask,
    )


def save_edge_refine_debug(
    before: Image.Image,
    result: SubjectEdgeRefineResult,
    *,
    output_dir: Path,
    stem: str,
    config: SubjectEdgeRefineConfig,
) -> None:
    if not config.debug_enabled:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    before.convert("RGBA").save(output_dir / f"{stem}_edge_refine_before.png", format="PNG")
    result.image.convert("RGBA").save(output_dir / f"{stem}_edge_refine_after.png", format="PNG")
    alpha = np.array(result.image.convert("RGBA").getchannel("A"))
    mask = np.zeros((alpha.shape[0], alpha.shape[1], 4), dtype=np.uint8)
    mask[alpha > int(config.effective_bbox_alpha_threshold)] = (0, 255, 0, 130)
    Image.fromarray(mask, "RGBA").save(output_dir / f"{stem}_edge_refine_mask.png", format="PNG")
    if result.edge_ring_mask is not None:
        ring = np.zeros((alpha.shape[0], alpha.shape[1], 4), dtype=np.uint8)
        ring[result.edge_ring_mask] = (255, 160, 0, 160)
        Image.fromarray(ring, "RGBA").save(output_dir / f"{stem}_edge_ring_mask.png", format="PNG")
        before_alpha = np.array(before.convert("RGBA").getchannel("A"), dtype=np.uint8)
        after_alpha = np.array(result.image.convert("RGBA").getchannel("A"), dtype=np.uint8)
        diff = np.clip((after_alpha.astype(np.int16) - before_alpha.astype(np.int16)) + 128, 0, 255).astype(np.uint8)
        Image.fromarray(diff, "L").save(output_dir / f"{stem}_edge_ring_alpha_before_after.png", format="PNG")
    if result.arm_edge_tighten_mask is not None:
        arm_mask = np.zeros((alpha.shape[0], alpha.shape[1], 4), dtype=np.uint8)
        arm_mask[result.arm_edge_tighten_mask] = (80, 180, 255, 170)
        Image.fromarray(arm_mask, "RGBA").save(output_dir / f"{stem}_arm_edge_tighten_mask.png", format="PNG")
