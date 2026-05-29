from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class TemporalSubjectFusionConfig:
    enabled: bool = True
    mode: str = "alpha_stability_fusion"
    min_frames: int = 4
    alignment_mode: str = "ecc_translation"
    alpha_vote_threshold: float = 0.6
    edge_consistency_weight: float = 0.35
    noise_component_min_area_ratio: float = 0.001
    fallback_to_single: bool = True
    debug_enabled: bool = True

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "TemporalSubjectFusionConfig":
        if not isinstance(raw, dict):
            return cls()
        values = {}
        for field_name in cls.__dataclass_fields__:
            if field_name in raw:
                values[field_name] = raw[field_name]
        return cls(**values)


@dataclass(frozen=True)
class TemporalFusionReport:
    alignment_success_count: int
    alpha_stable_ratio: float
    removed_temporal_noise_px: int
    fallback_reason: str | None = None


def _alpha_mask(image: Image.Image) -> np.ndarray:
    return (np.array(image.convert("RGBA").getchannel("A")) > 0).astype(np.uint8)


def _select_anchor_index(subjects: list[Image.Image]) -> int:
    areas = [int(np.count_nonzero(_alpha_mask(img))) for img in subjects]
    if not areas:
        return 0
    return int(np.argmax(np.array(areas)))


def _normalize_subject_size(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    rgba = image.convert("RGBA")
    if rgba.size == target_size:
        return rgba
    target_w, target_h = target_size
    scale = min(target_w / float(rgba.width), target_h / float(rgba.height))
    new_w = max(1, int(round(rgba.width * scale)))
    new_h = max(1, int(round(rgba.height * scale)))
    resized = rgba.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", target_size, (0, 0, 0, 0))
    offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)
    canvas.paste(resized, offset, resized)
    return canvas


def _align_to_anchor(
    anchor_mask: np.ndarray,
    target_rgba: np.ndarray,
    *,
    iterations: int = 30,
    eps: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, bool]:
    target_alpha = (target_rgba[:, :, 3] > 0).astype(np.uint8)
    if not np.any(target_alpha) or not np.any(anchor_mask):
        return target_rgba, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32), False

    warp = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, eps)

    try:
        cv2.findTransformECC(
            anchor_mask.astype(np.float32),
            target_alpha.astype(np.float32),
            warp,
            cv2.MOTION_TRANSLATION,
            criteria,
            None,
            1,
        )
        aligned = cv2.warpAffine(
            target_rgba,
            warp,
            (target_rgba.shape[1], target_rgba.shape[0]),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        return aligned, warp, True
    except cv2.error:
        return target_rgba, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32), False


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return mask
    keep = np.zeros_like(mask, dtype=bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if label == largest or area >= min_area:
            keep |= labels == label
    return keep


def _keep_soft_connected_to_core(core_mask: np.ndarray, soft_mask: np.ndarray) -> np.ndarray:
    if not np.any(core_mask):
        return core_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    near_core = cv2.dilate(core_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return core_mask | (soft_mask & near_core)


def _warp_mask_to_frame(mask: np.ndarray, warp: np.ndarray) -> np.ndarray:
    inv = np.array([[1.0, 0.0, -float(warp[0, 2])], [0.0, 1.0, -float(warp[1, 2])]], dtype=np.float32)
    alpha = mask.astype(np.uint8)
    if int(alpha.max()) <= 1:
        alpha = alpha * 255
    return cv2.warpAffine(
        alpha,
        inv,
        (mask.shape[1], mask.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def fuse_subjects_temporally(
    subjects: list[Image.Image | None],
    config: TemporalSubjectFusionConfig,
    *,
    debug_dir: Path | None = None,
    debug_stem: str = "",
) -> tuple[list[Image.Image | None], TemporalFusionReport]:
    valid_idx = [i for i, item in enumerate(subjects) if isinstance(item, Image.Image)]
    if not config.enabled or len(valid_idx) < max(1, int(config.min_frames)):
        return subjects, TemporalFusionReport(0, 0.0, 0, "insufficient_frames")

    valid_subjects = [subjects[i].convert("RGBA") for i in valid_idx]  # type: ignore[union-attr]
    anchor_local_idx = _select_anchor_index(valid_subjects)
    anchor_local = valid_subjects[anchor_local_idx]
    target_size = anchor_local.size
    valid_subjects = [_normalize_subject_size(img, target_size) for img in valid_subjects]
    anchor_local = valid_subjects[anchor_local_idx]
    anchor_mask = _alpha_mask(anchor_local)

    aligned_rgba: list[np.ndarray] = []
    warps: list[np.ndarray] = []
    success_count = 0
    for local_i, subject in enumerate(valid_subjects):
        arr = np.array(subject.convert("RGBA"))
        if local_i == anchor_local_idx:
            aligned_rgba.append(arr)
            warps.append(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32))
            success_count += 1
            continue
        aligned, warp, ok = _align_to_anchor(anchor_mask, arr)
        aligned_rgba.append(aligned)
        warps.append(warp)
        if ok:
            success_count += 1

    if success_count < int(config.min_frames):
        reason = "alignment_insufficient"
        if config.fallback_to_single:
            return subjects, TemporalFusionReport(success_count, 0.0, 0, reason)

    alpha_stack = np.stack([(img[:, :, 3] > 0).astype(np.float32) for img in aligned_rgba], axis=0)
    vote_ratio = np.mean(alpha_stack, axis=0)

    edge_stack = np.stack(
        [cv2.Canny((img[:, :, 3] > 0).astype(np.uint8) * 255, 60, 120).astype(np.float32) / 255.0 for img in aligned_rgba],
        axis=0,
    )
    edge_consistency = np.mean(edge_stack, axis=0)
    edge_consistency = cv2.GaussianBlur(edge_consistency, (0, 0), sigmaX=1.2, sigmaY=1.2)

    edge_w = float(np.clip(config.edge_consistency_weight, 0.0, 1.0))
    stability = np.clip(vote_ratio + edge_consistency * edge_w, 0.0, 1.0)
    core_thr = float(np.clip(config.alpha_vote_threshold, 0.0, 1.0))
    soft_thr = float(np.clip(min(core_thr, max(0.35, core_thr - 0.18)), 0.0, 1.0))
    core_mask = stability >= core_thr
    soft_mask = stability >= soft_thr
    fused_mask = _keep_soft_connected_to_core(core_mask, soft_mask)

    h, w = fused_mask.shape
    min_area = max(int(h * w * float(config.noise_component_min_area_ratio)), 24)
    before_px = int(np.count_nonzero(fused_mask))
    fused_mask = _remove_small_components(fused_mask, min_area=min_area)
    after_px = int(np.count_nonzero(fused_mask))
    removed_px = max(0, before_px - after_px)
    stable_ratio = float(after_px) / float(h * w)

    fused_alpha_anchor = cv2.GaussianBlur((fused_mask.astype(np.uint8) * 255), (0, 0), sigmaX=0.9, sigmaY=0.9)
    fused_alpha_anchor = np.where(fused_mask | (fused_alpha_anchor > 0), fused_alpha_anchor, 0).astype(np.uint8)

    if debug_dir is not None and config.debug_enabled:
        debug_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray((vote_ratio * 255).astype(np.uint8), "L").save(
            debug_dir / f"{debug_stem}_fusion_stability_map.png",
            format="PNG",
        )
        Image.fromarray(fused_alpha_anchor, "L").save(debug_dir / f"{debug_stem}_fusion_alpha_after.png", format="PNG")

    output = list(subjects)
    for local_i, global_i in enumerate(valid_idx):
        src_rgba = np.array(valid_subjects[local_i].convert("RGBA"))
        alpha_back = _warp_mask_to_frame(fused_alpha_anchor, warps[local_i])
        src_rgba[:, :, 3] = np.minimum(src_rgba[:, :, 3], alpha_back)
        output[global_i] = Image.fromarray(src_rgba, "RGBA")

    return output, TemporalFusionReport(
        alignment_success_count=success_count,
        alpha_stable_ratio=stable_ratio,
        removed_temporal_noise_px=removed_px,
        fallback_reason=None,
    )
