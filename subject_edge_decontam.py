from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SubjectEdgeDecontamConfig:
    enabled: bool = True
    edge_alpha_min: int = 12
    edge_alpha_max: int = 248
    boundary_outer_px: int = 4
    background_mode: str = "auto_ring"
    fixed_background_rgb: tuple[int, int, int] = (255, 255, 255)
    background_sample_max_alpha: int = 8
    min_background_samples: int = 24
    strength: float = 1.0
    debug_enabled: bool = True

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "SubjectEdgeDecontamConfig":
        if not isinstance(raw, dict):
            return cls()
        values: dict[str, Any] = {}
        for field_name in cls.__dataclass_fields__:
            if field_name not in raw:
                continue
            value = raw[field_name]
            if field_name == "fixed_background_rgb" and isinstance(value, (list, tuple)) and len(value) == 3:
                values[field_name] = tuple(int(v) for v in value)
            else:
                values[field_name] = value
        if "edge_band_px" in raw and "boundary_outer_px" not in raw:
            values["boundary_outer_px"] = int(raw["edge_band_px"])
        if "decontam_strength" in raw and "strength" not in raw:
            values["strength"] = float(raw["decontam_strength"])
        return cls(**values)


@dataclass(frozen=True)
class SubjectEdgeDecontamResult:
    image: Image.Image
    edge_pixel_count: int
    estimated_background_rgb: tuple[int, int, int]
    mean_edge_luma_delta: float
    edge_mask: np.ndarray | None = None


def _morph(mask: np.ndarray, operation: int, radius: int) -> np.ndarray:
    if int(radius) <= 0:
        return mask.astype(bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(radius) * 2 + 1, int(radius) * 2 + 1))
    return cv2.morphologyEx(mask.astype(np.uint8), operation, kernel).astype(bool)


def _build_edge_mask(alpha: np.ndarray, config: SubjectEdgeDecontamConfig) -> np.ndarray:
    subject = alpha > int(config.edge_alpha_min)
    if not np.any(subject):
        return np.zeros_like(alpha, dtype=bool)
    boundary = subject & ~_morph(subject, cv2.MORPH_ERODE, 1)
    outer_ring = _morph(subject, cv2.MORPH_DILATE, int(config.boundary_outer_px)) & ~subject
    semi_transparent = (alpha >= int(config.edge_alpha_min)) & (alpha <= int(config.edge_alpha_max))
    return (boundary | outer_ring) & semi_transparent


def estimate_background_rgb(
    rgb: np.ndarray,
    alpha: np.ndarray,
    config: SubjectEdgeDecontamConfig,
    *,
    edge_mask: np.ndarray | None = None,
) -> tuple[int, int, int]:
    mode = str(config.background_mode).lower()
    if mode == "fixed":
        return tuple(int(v) for v in config.fixed_background_rgb)

    subject = alpha > int(config.edge_alpha_min)
    if edge_mask is None:
        edge_mask = _build_edge_mask(alpha, config)
    sample_ring = _morph(subject, cv2.MORPH_DILATE, int(config.boundary_outer_px)) & ~subject
    sample_mask = sample_ring & (alpha <= int(config.background_sample_max_alpha))
    if int(np.count_nonzero(sample_mask)) < int(config.min_background_samples):
        sample_mask = sample_ring
    if int(np.count_nonzero(sample_mask)) < int(config.min_background_samples) and np.any(edge_mask):
        sample_mask = edge_mask & (alpha <= int(config.edge_alpha_max))
    if int(np.count_nonzero(sample_mask)) < int(config.min_background_samples):
        return tuple(int(v) for v in config.fixed_background_rgb)

    samples = rgb[sample_mask].reshape(-1, 3).astype(np.float32)
    median = np.median(samples, axis=0)
    return int(median[0]), int(median[1]), int(median[2])


def decontaminate_subject_edges(
    image: Image.Image,
    config: SubjectEdgeDecontamConfig,
) -> SubjectEdgeDecontamResult:
    rgba = image.convert("RGBA")
    if not config.enabled:
        return SubjectEdgeDecontamResult(rgba, 0, config.fixed_background_rgb, 0.0, None)

    arr = np.array(rgba)
    alpha = arr[:, :, 3]
    edge_mask = _build_edge_mask(alpha, config)
    edge_pixel_count = int(np.count_nonzero(edge_mask))
    if edge_pixel_count <= 0:
        return SubjectEdgeDecontamResult(rgba, 0, config.fixed_background_rgb, 0.0, edge_mask)

    rgb = arr[:, :, :3].astype(np.float32)
    bg = np.array(estimate_background_rgb(rgb, alpha, config, edge_mask=edge_mask), dtype=np.float32)
    alpha_f = alpha.astype(np.float32) / 255.0
    alpha_safe = np.maximum(alpha_f, 1.0 / 255.0)
    foreground = (rgb - bg.reshape(1, 1, 3) * (1.0 - alpha_f[..., None])) / alpha_safe[..., None]
    foreground = np.clip(foreground, 0.0, 255.0)

    strength = float(np.clip(config.strength, 0.0, 1.0))
    blended = rgb.copy()
    if strength >= 1.0:
        blended[edge_mask] = foreground[edge_mask]
    else:
        blended[edge_mask] = rgb[edge_mask] * (1.0 - strength) + foreground[edge_mask] * strength

    core_keep = alpha >= int(config.edge_alpha_max)
    blended[core_keep] = rgb[core_keep]

    before_luma = np.mean(rgb[edge_mask])
    after_luma = np.mean(blended[edge_mask])
    arr[:, :, :3] = blended.astype(np.uint8)
    return SubjectEdgeDecontamResult(
        Image.fromarray(arr, "RGBA"),
        edge_pixel_count,
        (int(bg[0]), int(bg[1]), int(bg[2])),
        float(before_luma - after_luma),
        edge_mask,
    )


def save_decontam_debug(
    before: Image.Image,
    result: SubjectEdgeDecontamResult,
    *,
    output_dir: Path,
    stem: str,
    edge_mask: np.ndarray | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    before.convert("RGBA").save(output_dir / f"{stem}_decontam_before.png", format="PNG")
    result.image.convert("RGBA").save(output_dir / f"{stem}_decontam_after.png", format="PNG")
    if edge_mask is not None:
        overlay = np.zeros((edge_mask.shape[0], edge_mask.shape[1], 4), dtype=np.uint8)
        overlay[edge_mask] = (255, 80, 80, 170)
        Image.fromarray(overlay, "RGBA").save(output_dir / f"{stem}_decontam_edge_mask.png", format="PNG")
