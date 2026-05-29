from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from ali_segment_service import AliSegmentService, _create_pipeline
from app_state import APP_STATE, BASE_DIR, OUTPUT_DIR
from background_manager import get_background_items
from config_manager import load_config
from image_composer import _place_subject_on_background
from modnet_matting_service import (
    AlphaConstraintResult,
    ModnetMattingError,
    ModnetMattingService,
    apply_instance_alpha_constraints,
)
from slogan_manager import get_rotation_snapshot
from subject_edge_refine import SubjectEdgeRefineConfig, refine_subject_edge, save_edge_refine_debug
from subject_instance_segmentation import (
    InstanceSegmentationConfig,
    InstanceSegmentationResult,
    SubjectInstanceSegmenter,
    cutout_from_instance_mask,
)
from subject_temporal_fusion import TemporalSubjectFusionConfig, fuse_subjects_temporally
from text_renderer import draw_slogan
from matanyone_service import (
    MatAnyoneConstraintConfig,
    MatAnyoneConstraintResult,
    MatAnyoneService,
    apply_matanyone_alpha_constraints,
    build_matanyone_initial_mask,
)
from vitmatte_service import (
    VitmatteDirectionalRelaxConfig,
    VitmatteDirectionalRelaxResult,
    VitmatteService,
    build_vitmatte_directional_constraints,
)
from yolo_seg_aliyun_service import (
    YoloSegAliyunConfig,
    YoloSegAliyunService,
    apply_instance_alpha_to_roi,
)
from modelscope_universal_matting_service import (
    MODELSCOPE_UNIVERSAL_MODEL_ID,
    ModelScopeUniversalMattingService,
)

try:
    from gpupixel_beauty_service import apply_gpupixel_preset, compute_color_beauty_metrics
except ModuleNotFoundError:
    apply_gpupixel_preset = None
    compute_color_beauty_metrics = None


DEFAULT_GROUPS = [
    "8032532334c940d28cf78782fc2d43b3",
    "9595dd5a6d504901a8f6911a9a951353",
]


def trimap_config_to_dict(config: InstanceSegmentationConfig) -> dict[str, Any]:
    return {
        "sure_fg_erode_px": int(config.sure_fg_erode_px),
        "subject_unknown_dilate_px": int(config.subject_unknown_dilate_px),
        "visitor_bg_dilate_px": int(config.visitor_bg_dilate_px),
    }


def build_instance_config(
    *,
    yolo_seg_model_path: str,
    sure_fg_erode_px: int | None = None,
    subject_unknown_dilate_px: int | None = None,
    visitor_bg_dilate_px: int | None = None,
) -> InstanceSegmentationConfig:
    config = InstanceSegmentationConfig(model_path=yolo_seg_model_path)
    overrides: dict[str, int] = {}
    if sure_fg_erode_px is not None:
        overrides["sure_fg_erode_px"] = int(sure_fg_erode_px)
    if subject_unknown_dilate_px is not None:
        overrides["subject_unknown_dilate_px"] = int(subject_unknown_dilate_px)
    if visitor_bg_dilate_px is not None:
        overrides["visitor_bg_dilate_px"] = int(visitor_bg_dilate_px)
    if not overrides:
        return config
    return InstanceSegmentationConfig(**{**asdict(config), **overrides})


def aggregate_branch_metrics(summary: dict[str, Any], branch: str) -> dict[str, float]:
    metrics_key = f"{branch}_metrics"
    visitor_vals: list[float] = []
    core_vals: list[float] = []
    fragment_vals: list[float] = []
    foreground_vals: list[float] = []
    for group_data in summary.get("groups", {}).values():
        for frame in group_data.get("frames", []):
            metrics = frame.get(metrics_key)
            if not metrics:
                continue
            visitor_vals.append(float(metrics["visitor_residual_ratio"]))
            core_vals.append(float(metrics["subject_core_missing_ratio"]))
            fragment_vals.append(float(metrics["fragment_count"]))
            foreground_vals.append(float(metrics["foreground_px"]))
    if not visitor_vals:
        return {
            "visitor_residual_ratio_avg": 1.0,
            "visitor_residual_ratio_max": 1.0,
            "subject_core_missing_ratio_avg": 1.0,
            "subject_core_missing_ratio_max": 1.0,
            "fragment_count_avg": 99.0,
            "fragment_count_max": 99.0,
            "foreground_px_avg": 0.0,
            "frame_count": 0.0,
        }
    return {
        "visitor_residual_ratio_avg": sum(visitor_vals) / len(visitor_vals),
        "visitor_residual_ratio_max": max(visitor_vals),
        "subject_core_missing_ratio_avg": sum(core_vals) / len(core_vals),
        "subject_core_missing_ratio_max": max(core_vals),
        "fragment_count_avg": sum(fragment_vals) / len(fragment_vals),
        "fragment_count_max": max(fragment_vals),
        "foreground_px_avg": sum(foreground_vals) / len(foreground_vals),
        "frame_count": float(len(visitor_vals)),
    }


def aggregate_modnet_metrics(summary: dict[str, Any]) -> dict[str, float]:
    return aggregate_branch_metrics(summary, "modnet")


def aggregate_matanyone_detail_metrics(summary: dict[str, Any]) -> dict[str, float]:
    ratio_values: list[float] = []
    pixel_values: list[float] = []
    hair_outside_values: list[float] = []
    hair_removed_values: list[float] = []
    hair_retained_values: list[float] = []
    protected_core_missing_values: list[float] = []
    for group_data in summary.get("groups", {}).values():
        for frame in group_data.get("frames", []):
            metrics = frame.get("matanyone_metrics")
            if not metrics:
                continue
            ratio_values.append(float(metrics["outside_subject_alpha_ratio"]))
            pixel_values.append(float(metrics["outside_subject_alpha_px"]))
            hair_outside_values.append(float(metrics.get("right_hair_outside_alpha_px", 0)))
            hair_removed_values.append(float(metrics.get("right_hair_removed_alpha_px", 0)))
            hair_retained_values.append(float(metrics.get("right_hair_retained_alpha_px", 0)))
            protected_core_missing_values.append(float(metrics.get("subject_core_missing_ratio_excluding_hair_rejudge", 0)))
    if not ratio_values:
        return {
            "outside_subject_alpha_ratio_avg": 1.0,
            "outside_subject_alpha_ratio_max": 1.0,
            "outside_subject_alpha_px_avg": 0.0,
            "outside_subject_alpha_px_max": 0.0,
            "right_hair_outside_alpha_px_avg": 0.0,
            "right_hair_removed_alpha_px_avg": 0.0,
            "right_hair_retained_alpha_px_avg": 0.0,
            "subject_core_missing_ratio_excluding_hair_rejudge_avg": 1.0,
            "subject_core_missing_ratio_excluding_hair_rejudge_max": 1.0,
            "frame_count": 0.0,
        }
    return {
        "outside_subject_alpha_ratio_avg": sum(ratio_values) / len(ratio_values),
        "outside_subject_alpha_ratio_max": max(ratio_values),
        "outside_subject_alpha_px_avg": sum(pixel_values) / len(pixel_values),
        "outside_subject_alpha_px_max": max(pixel_values),
        "right_hair_outside_alpha_px_avg": sum(hair_outside_values) / len(hair_outside_values),
        "right_hair_removed_alpha_px_avg": sum(hair_removed_values) / len(hair_removed_values),
        "right_hair_retained_alpha_px_avg": sum(hair_retained_values) / len(hair_retained_values),
        "subject_core_missing_ratio_excluding_hair_rejudge_avg": sum(protected_core_missing_values)
        / len(protected_core_missing_values),
        "subject_core_missing_ratio_excluding_hair_rejudge_max": max(protected_core_missing_values),
        "frame_count": float(len(ratio_values)),
    }


def aggregate_gpupixel_metrics(summary: dict[str, Any]) -> dict[str, float]:
    red_cast: list[float] = []
    red_delta: list[float] = []
    luminance_gain: list[float] = []
    naturalness: list[float] = []
    edge_artifact_delta: list[float] = []
    fallback_count = 0
    frame_count = 0
    for group_data in summary.get("groups", {}).values():
        for frame in group_data.get("frames", []):
            metrics = frame.get("gpupixel_metrics")
            if not metrics:
                continue
            frame_count += 1
            if frame.get("gpupixel_fallback"):
                fallback_count += 1
            red_cast.append(float(metrics.get("red_cast_score", 0.0)))
            red_delta.append(float(metrics.get("red_cast_delta", 0.0)))
            luminance_gain.append(float(metrics.get("luminance_gain", 0.0)))
            naturalness.append(float(metrics.get("skin_naturalness_proxy", 0.0)))
            edge_artifact_delta.append(float(metrics.get("edge_artifact_delta", 0.0)))
    if frame_count == 0:
        return {
            "frame_count": 0.0,
            "fallback_ratio": 1.0,
            "red_cast_score_avg": 0.0,
            "red_cast_delta_avg": 0.0,
            "luminance_gain_avg": 0.0,
            "skin_naturalness_proxy_avg": 0.0,
            "edge_artifact_delta_avg": 0.0,
        }
    return {
        "frame_count": float(frame_count),
        "fallback_ratio": float(fallback_count) / float(frame_count),
        "red_cast_score_avg": sum(red_cast) / frame_count,
        "red_cast_delta_avg": sum(red_delta) / frame_count,
        "luminance_gain_avg": sum(luminance_gain) / frame_count,
        "skin_naturalness_proxy_avg": sum(naturalness) / frame_count,
        "edge_artifact_delta_avg": sum(edge_artifact_delta) / frame_count,
    }


def aggregate_comparison_metrics(summary: dict[str, Any]) -> dict[str, dict[str, float]]:
    metrics = {
        "current_aliyun": aggregate_branch_metrics(summary, "current_aliyun"),
        "yolo_seg_mask": aggregate_branch_metrics(summary, "mask"),
        "yolo_seg_modnet": aggregate_branch_metrics(summary, "modnet"),
        "yolo_seg_aliyun": aggregate_branch_metrics(summary, "yolo_seg_aliyun"),
        "yolo_seg_vitmatte": aggregate_branch_metrics(summary, "vitmatte"),
        "modelscope_universal_matting": aggregate_branch_metrics(summary, "modelscope_universal"),
        "matanyone": aggregate_branch_metrics(summary, "matanyone"),
    }
    matanyone_config = summary.get("matanyone_constraint_config", {})
    if matanyone_config.get("initial_mask_mode") == "subject_instance_mask":
        metrics["matanyone_subject_constrained"] = aggregate_branch_metrics(summary, "matanyone")
    elif matanyone_config.get("initial_mask_mode") == "bbox":
        metrics["matanyone_bbox_baseline"] = aggregate_branch_metrics(summary, "matanyone")
    metrics["matanyone_edge_detail"] = aggregate_matanyone_detail_metrics(summary)
    vitmatte_config = summary.get("vitmatte_directional_relax_config", {})
    if vitmatte_config.get("mode") == "contact_local":
        metrics["vitmatte_contact_local"] = aggregate_branch_metrics(summary, "vitmatte")
    metrics["gpupixel_color_beauty_eval"] = aggregate_branch_metrics(summary, "gpupixel_color_beauty_eval")
    metrics["gpupixel_color_beauty_detail"] = aggregate_gpupixel_metrics(summary)
    return metrics


def _compose_final(subject: Image.Image, background_item: dict[str, Any], slogan: str, slogan_row: int | None, subject_bbox: tuple[int, int, int, int] | None = None) -> Image.Image:
    output_cfg = APP_STATE["config"]["output"]
    target_size = (int(output_cfg["width"]), int(output_cfg["height"]))
    background = _place_subject_on_background(subject, background_item, target_size, subject_bbox=subject_bbox)
    return draw_slogan(background, slogan, background_item, slogan_row)


def _component_count(alpha: np.ndarray, threshold: int = 16) -> int:
    mask = (alpha > threshold).astype(np.uint8)
    count, _labels, _stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return max(count - 1, 0)


def reapply_sure_foreground_alpha(image: Image.Image, result: InstanceSegmentationResult) -> Image.Image:
    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    arr[result.sure_foreground.astype(bool), 3] = 255
    return Image.fromarray(arr, "RGBA")


def restore_constrained_alpha_after_refine(image: Image.Image, result: InstanceSegmentationResult) -> Image.Image:
    restored = reapply_sure_foreground_alpha(image, result)
    arr = np.array(restored.convert("RGBA"))
    arr[result.sure_background.astype(bool), 3] = 0
    return Image.fromarray(arr, "RGBA")


def apply_modelscope_universal_constraints(image: Image.Image, result: InstanceSegmentationResult) -> Image.Image:
    rgba = image.convert("RGBA")
    if rgba.size != result.image_size:
        rgba = rgba.resize(result.image_size, Image.Resampling.BILINEAR)
    arr = np.array(rgba)
    arr[result.sure_background.astype(bool), 3] = 0
    arr[result.sure_foreground.astype(bool), 3] = 255
    return Image.fromarray(arr, "RGBA")


def _visitor_mask(result: InstanceSegmentationResult) -> np.ndarray:
    mask = np.zeros((result.image_size[1], result.image_size[0]), dtype=bool)
    for visitor in result.visitors:
        mask |= visitor.mask.astype(bool)
    return mask


def _mask_in_output_space(
    mask: np.ndarray,
    output_size: tuple[int, int],
    roi_box: tuple[int, int, int, int] | None,
) -> np.ndarray:
    mapped = mask.astype(np.uint8)
    if roi_box is not None:
        left, top, right, bottom = roi_box
        mapped = mapped[top:bottom, left:right]
    if mapped.shape != (output_size[1], output_size[0]):
        mapped = cv2.resize(mapped, output_size, interpolation=cv2.INTER_NEAREST)
    return mapped.astype(bool)


def compute_matting_metrics(
    image: Image.Image,
    result: InstanceSegmentationResult,
    *,
    roi_box: tuple[int, int, int, int] | None = None,
) -> dict[str, Any]:
    alpha = np.array(image.convert("RGBA").getchannel("A"), dtype=np.uint8)
    visitor = _mask_in_output_space(_visitor_mask(result), image.size, roi_box)
    sure_fg = _mask_in_output_space(result.sure_foreground, image.size, roi_box)
    visitor_px = int(np.count_nonzero(visitor))
    fg_px = int(np.count_nonzero(sure_fg))
    visitor_alpha = int(np.count_nonzero((alpha > 16) & visitor))
    missing_core = int(np.count_nonzero((alpha <= 240) & sure_fg))
    return {
        "visitor_residual_ratio": float(visitor_alpha) / float(max(visitor_px, 1)),
        "subject_core_missing_ratio": float(missing_core) / float(max(fg_px, 1)),
        "fragment_count": _component_count(alpha, threshold=16),
        "foreground_px": int(np.count_nonzero(alpha > 16)),
    }


def compute_vitmatte_directional_metrics(
    image: Image.Image,
    original_result: InstanceSegmentationResult,
    constrained_result: InstanceSegmentationResult,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        compute_matting_metrics(image, constrained_result),
        compute_matting_metrics(image, original_result),
    )


def compute_matanyone_metrics(
    constraint: MatAnyoneConstraintResult,
    result: InstanceSegmentationResult,
) -> dict[str, Any]:
    metrics = compute_matting_metrics(constraint.image, result)
    alpha = np.array(constraint.image.getchannel("A"), dtype=np.uint8)
    protected_core = result.sure_foreground.astype(bool) & ~constraint.masks.hair_inner_rejudge
    protected_missing = int(np.count_nonzero(protected_core & (alpha <= 16)))
    metrics.update(
        {
            "raw_foreground_px": constraint.raw_foreground_px,
            "constrained_foreground_px": constraint.constrained_foreground_px,
            "outside_subject_alpha_px": constraint.outside_subject_alpha_px,
            "outside_subject_alpha_ratio": constraint.outside_subject_alpha_ratio,
            "soft_band_alpha_px": constraint.soft_band_alpha_px,
            "right_hair_outside_alpha_px": constraint.right_hair_outside_alpha_px,
            "right_hair_removed_alpha_px": constraint.right_hair_removed_alpha_px,
            "right_hair_retained_alpha_px": constraint.right_hair_retained_alpha_px,
            "subject_core_missing_ratio_excluding_hair_rejudge": float(protected_missing)
            / float(max(int(np.count_nonzero(protected_core)), 1)),
        }
    )
    return metrics


def _save_debug_alpha(output_dir: Path, stem: str, constraint: AlphaConstraintResult) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(constraint.raw_alpha, "L").save(output_dir / f"{stem}_modnet_raw_alpha.png", format="PNG")
    Image.fromarray(constraint.constrained_alpha, "L").save(output_dir / f"{stem}_modnet_constrained_alpha.png", format="PNG")


def _save_vitmatte_directional_debug(
    output_dir: Path,
    stem: str,
    relaxation: VitmatteDirectionalRelaxResult,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(relaxation.result.trimap.astype(np.uint8), "L").save(
        output_dir / f"{stem}_vitmatte_directional_trimap.png",
        format="PNG",
    )
    Image.fromarray((relaxation.directional_unknown.astype(np.uint8) * 255), "L").save(
        output_dir / f"{stem}_vitmatte_directional_unknown.png",
        format="PNG",
    )
    Image.fromarray((relaxation.contact_zone.astype(np.uint8) * 255), "L").save(
        output_dir / f"{stem}_vitmatte_contact_zone.png",
        format="PNG",
    )


def _save_matanyone_debug(
    output_dir: Path,
    stem: str,
    constraint: MatAnyoneConstraintResult,
    *,
    refined_image: Image.Image | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(constraint.raw_alpha, "L").save(output_dir / f"{stem}_matanyone_raw_alpha.png", format="PNG")
    Image.fromarray((constraint.masks.allowed_support.astype(np.uint8) * 255), "L").save(
        output_dir / f"{stem}_matanyone_allowed_support.png", format="PNG"
    )
    Image.fromarray((constraint.masks.forced_background.astype(np.uint8) * 255), "L").save(
        output_dir / f"{stem}_matanyone_forced_background.png", format="PNG"
    )
    Image.fromarray((constraint.masks.hair_region.astype(np.uint8) * 255), "L").save(
        output_dir / f"{stem}_matanyone_right_hair_region.png", format="PNG"
    )
    Image.fromarray((constraint.masks.hair_inner_rejudge.astype(np.uint8) * 255), "L").save(
        output_dir / f"{stem}_matanyone_right_hair_inner_rejudge.png", format="PNG"
    )
    Image.fromarray((constraint.masks.hair_outer_support.astype(np.uint8) * 255), "L").save(
        output_dir / f"{stem}_matanyone_right_hair_outer_support.png", format="PNG"
    )
    hair_comparison = Image.merge(
        "RGB",
        (
            Image.fromarray(constraint.raw_alpha, "L"),
            Image.fromarray(constraint.constrained_alpha, "L"),
            Image.fromarray((constraint.masks.hair_inner_rejudge.astype(np.uint8) * 255), "L"),
        ),
    )
    hair_comparison.save(output_dir / f"{stem}_matanyone_right_hair_alpha_before_after.png", format="PNG")
    Image.fromarray(constraint.constrained_alpha, "L").save(
        output_dir / f"{stem}_matanyone_constrained_alpha.png", format="PNG"
    )
    if refined_image is not None:
        refined_image.getchannel("A").save(output_dir / f"{stem}_matanyone_edge_refined_alpha.png", format="PNG")


def _save_sheet(items: list[tuple[Image.Image, str]], output_path: Path) -> None:
    thumb_w, thumb_h, gap = 240, 330, 14
    canvas = Image.new("RGB", (len(items) * thumb_w + (len(items) + 1) * gap, thumb_h + 46), (244, 238, 224))
    draw = ImageDraw.Draw(canvas)
    for idx, (image, label) in enumerate(items):
        x = gap + idx * (thumb_w + gap)
        y = 34
        fit = ImageOps.contain(image.convert("RGBA"), (thumb_w, thumb_h))
        tile = Image.new("RGBA", (thumb_w, thumb_h), (255, 255, 255, 255))
        tile.paste(fit, ((thumb_w - fit.width) // 2, (thumb_h - fit.height) // 2), fit)
        canvas.paste(tile.convert("RGB"), (x, y))
        draw.text((x + 6, 9), label, fill=(25, 25, 25))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="JPEG", quality=92)


def _sheet_background(size: tuple[int, int], cell: int = 16) -> Image.Image:
    background = Image.new("RGBA", size, (68, 68, 68, 255))
    draw = ImageDraw.Draw(background)
    for y in range(0, size[1], cell):
        for x in range(0, size[0], cell):
            if ((x // cell) + (y // cell)) % 2 == 0:
                draw.rectangle((x, y, x + cell - 1, y + cell - 1), fill=(112, 112, 112, 255))
    return background


def generate_comparison_sheet(*, out_root: Path, group: str, kind: str, output_path: Path) -> bool:
    branches = (
        ("current_aliyun", "CURRENT ALIYUN"),
        ("gpupixel_color_beauty_eval", "GPUPIXEL + CURRENT"),
        ("yolo_seg_mask", "YOLO-SEG MASK"),
        ("yolo_seg_modnet", "YOLO-SEG + MODNET"),
        ("yolo_seg_aliyun", "YOLO-SEG + ALIYUN"),
        ("yolo_seg_vitmatte", "YOLO-SEG + ViTMatte"),
        ("modelscope_universal_matting", "MODELSCOPE UNIVERSAL"),
        ("matanyone_bbox_baseline", "MatAnyone2 BBOX"),
        ("matanyone_subject_constrained", "MatAnyone2 CONSTRAINED"),
        ("matanyone", "MatAnyone2"),
    )
    available: list[tuple[str, str, list[Path]]] = []
    for branch, title in branches:
        paths = [out_root / branch / kind / group / f"{group}_{index}.{('png' if kind == 'cutouts' else 'jpg')}" for index in range(1, 5)]
        if any(path.exists() for path in paths):
            available.append((branch, title, paths))
    if not available:
        return False

    tile_w, tile_h = 230, 250
    left, gap, top, row_gap = 146, 14, 36, 40
    width = left + 4 * tile_w + 5 * gap
    height = top + len(available) * (tile_h + row_gap) + gap
    canvas = Image.new("RGB", (width, height), (244, 238, 224))
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 10), f"{group} - {kind}", fill=(25, 25, 25))
    for row, (_branch, title, paths) in enumerate(available):
        y = top + row * (tile_h + row_gap)
        draw.text((gap, y + tile_h // 2 - 8), title, fill=(25, 25, 25))
        for index, path in enumerate(paths, start=1):
            x = left + (index - 1) * (tile_w + gap)
            tile = _sheet_background((tile_w, tile_h)) if kind == "cutouts" else Image.new("RGBA", (tile_w, tile_h), (255, 255, 255, 255))
            if path.exists():
                with Image.open(path) as opened:
                    fit = ImageOps.contain(opened.convert("RGBA"), (tile_w, tile_h))
                tile.alpha_composite(fit, ((tile_w - fit.width) // 2, (tile_h - fit.height) // 2))
            canvas.paste(tile.convert("RGB"), (x, y))
            draw.text((x + 6, y + tile_h + 10), str(index), fill=(25, 25, 25))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="JPEG", quality=94)
    return True


def segment_current_aliyun_cutout(
    service: Any,
    capture_path: Path,
    output_path: Path,
) -> Image.Image:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = service.segment_image_file(capture_path, output_path)
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    if not image.getbbox():
        raise ValueError("current_aliyun segment result is empty.")
    image.save(output_path, format="PNG")
    return image


def _process_group(
    group: str,
    captures_dir: Path,
    out_root: Path,
    segmenter: SubjectInstanceSegmenter,
    modnet: ModnetMattingService | None,
    current_aliyun: Any | None,
    include_gpupixel_color_beauty_eval: bool,
    gpupixel_preset: str,
    yolo_seg_aliyun: YoloSegAliyunService | None,
    modelscope_universal: ModelScopeUniversalMattingService | None,
    vitmatte: VitmatteService | None,
    vitmatte_directional_cfg: VitmatteDirectionalRelaxConfig,
    matanyone: MatAnyoneService | None,
    matanyone_cfg: MatAnyoneConstraintConfig,
    edge_cfg: SubjectEdgeRefineConfig,
    temporal_cfg: TemporalSubjectFusionConfig,
    background_item: dict[str, Any],
    slogan: str,
    slogan_row: int | None,
) -> dict[str, Any]:
    group_summary: dict[str, Any] = {"frames": [], "matting_available": modnet is not None}
    mask_cutouts: list[Image.Image] = []
    modnet_cutouts: list[Image.Image] = []
    current_aliyun_cutouts: list[Image.Image] = []
    gpupixel_cutouts: list[Image.Image] = []
    yolo_seg_aliyun_cutouts: list[Image.Image] = []
    modelscope_universal_cutouts: list[Image.Image] = []
    vitmatte_cutouts: list[Image.Image] = []
    matanyone_cutouts: list[Image.Image] = []
    frame_bboxes: list[tuple[int, int, int, int]] = []
    frame_results: list[InstanceSegmentationResult | None] = [None] * 4
    frame_summaries: list[dict[str, Any] | None] = [None] * 4

    for index in range(1, 5):
        stem = f"{group}_{index}"
        capture_path = captures_dir / f"{stem}.jpg"
        frame_summary: dict[str, Any] = {"stem": stem}
        frame_summaries[index - 1] = frame_summary
        t0 = time.perf_counter()
        result = segmenter.segment(capture_path, stem)
        frame_summary["instance_elapsed_seconds"] = time.perf_counter() - t0
        if result is None:
            frame_summary["ok"] = False
            frame_summary["error"] = "instance_segmentation_failed"
            group_summary["frames"].append(frame_summary)
            continue
        frame_results[index - 1] = result

        left, top, right, bottom = result.selected.bbox
        frame_bboxes.append((int(left), int(top), int(right), int(bottom)))

        mask_cutout = cutout_from_instance_mask(capture_path, result.selected)
        mask_refined = refine_subject_edge(mask_cutout, edge_cfg).image if edge_cfg.enabled else mask_cutout
        mask_cutouts.append(mask_refined)
        frame_summary["mask_metrics"] = compute_matting_metrics(mask_refined, result)
        mask_dir = out_root / "yolo_seg_mask" / "cutouts" / group
        mask_dir.mkdir(parents=True, exist_ok=True)
        mask_refined.save(mask_dir / f"{stem}.png", format="PNG")

        if modnet is not None:
            try:
                mt0 = time.perf_counter()
                constraint = modnet.matte_image_file(capture_path, result)
                modnet_image = constraint.image
                _save_debug_alpha(out_root / "debug", stem, constraint)
                refine_result = refine_subject_edge(modnet_image, edge_cfg) if edge_cfg.enabled else None
                if refine_result is not None:
                    save_edge_refine_debug(
                        modnet_image,
                        refine_result,
                        output_dir=out_root / "debug",
                        stem=f"{stem}_modnet",
                        config=edge_cfg,
                    )
                    modnet_image = refine_result.image
                modnet_image = reapply_sure_foreground_alpha(modnet_image, result)
                modnet_cutouts.append(modnet_image)
                modnet_dir = out_root / "yolo_seg_modnet" / "cutouts" / group
                modnet_dir.mkdir(parents=True, exist_ok=True)
                modnet_image.save(modnet_dir / f"{stem}.png", format="PNG")
                frame_summary["modnet_elapsed_seconds"] = time.perf_counter() - mt0
                frame_summary["modnet_metrics"] = compute_matting_metrics(modnet_image, result)
                frame_summary["constraint"] = {
                    "forced_foreground_px": constraint.forced_foreground_px,
                    "forced_background_px": constraint.forced_background_px,
                    "unknown_px": constraint.unknown_px,
                }
            except Exception as exc:
                frame_summary["modnet_error"] = str(exc)

        if current_aliyun is not None:
            try:
                at0 = time.perf_counter()
                aliyun_dir = out_root / "current_aliyun" / "cutouts" / group
                aliyun_image = segment_current_aliyun_cutout(
                    current_aliyun,
                    capture_path,
                    aliyun_dir / f"{stem}.png",
                )
                current_aliyun_cutouts.append(aliyun_image)
                frame_summary["current_aliyun_elapsed_seconds"] = time.perf_counter() - at0
                location = getattr(current_aliyun, "last_subject_location", None)
                roi_box = location.roi_box if location is not None else None
                frame_summary["current_aliyun_metrics"] = compute_matting_metrics(
                    aliyun_image,
                    result,
                    roi_box=roi_box,
                )
            except Exception as exc:
                frame_summary["current_aliyun_error"] = str(exc)

        if include_gpupixel_color_beauty_eval and current_aliyun is not None:
            try:
                if apply_gpupixel_preset is None or compute_color_beauty_metrics is None:
                    raise RuntimeError("gpupixel_beauty_service_not_available")
                gt0 = time.perf_counter()
                gpupixel_capture_dir = out_root / "gpupixel_color_beauty_eval" / "captures" / group
                gpupixel_capture_dir.mkdir(parents=True, exist_ok=True)
                gpupixel_result = apply_gpupixel_preset(Image.open(capture_path), preset_name=gpupixel_preset)
                gpupixel_capture_path = gpupixel_capture_dir / f"{stem}.jpg"
                gpupixel_result.image.convert("RGB").save(gpupixel_capture_path, format="JPEG", quality=95)
                debug_source_dir = out_root / "gpupixel_color_beauty_eval" / "source" / group
                debug_source_dir.mkdir(parents=True, exist_ok=True)
                Image.open(capture_path).convert("RGB").save(debug_source_dir / f"{stem}.jpg", format="JPEG", quality=95)

                branch_dir = out_root / "gpupixel_color_beauty_eval" / "cutouts" / group
                gpupixel_cutout = segment_current_aliyun_cutout(
                    current_aliyun,
                    gpupixel_capture_path,
                    branch_dir / f"{stem}.png",
                )
                refine_result = refine_subject_edge(gpupixel_cutout, edge_cfg) if edge_cfg.enabled else None
                if refine_result is not None:
                    save_edge_refine_debug(
                        gpupixel_cutout,
                        refine_result,
                        output_dir=out_root / "debug",
                        stem=f"{stem}_gpupixel",
                        config=edge_cfg,
                    )
                    gpupixel_cutout = refine_result.image
                    gpupixel_cutout.save(branch_dir / f"{stem}.png", format="PNG")
                gpupixel_cutouts.append(gpupixel_cutout)
                baseline_metrics = frame_summary.get("current_aliyun_metrics")
                gpupixel_metrics = compute_color_beauty_metrics(
                    Image.open(capture_path).convert("RGB"),
                    gpupixel_result.image,
                )
                gpupixel_cutout_metrics = compute_matting_metrics(gpupixel_cutout, result)
                gpupixel_metrics.update(gpupixel_cutout_metrics)
                if baseline_metrics:
                    gpupixel_metrics["edge_artifact_delta"] = float(
                        gpupixel_cutout_metrics["fragment_count"] - baseline_metrics.get("fragment_count", 0)
                    )
                else:
                    gpupixel_metrics["edge_artifact_delta"] = 0.0
                frame_summary["gpupixel_elapsed_seconds"] = time.perf_counter() - gt0
                frame_summary["gpupixel_metrics"] = gpupixel_metrics
                frame_summary["gpupixel_color_beauty_eval_metrics"] = gpupixel_cutout_metrics
                frame_summary["gpupixel_preset"] = gpupixel_result.preset_name
                frame_summary["gpupixel_fallback"] = gpupixel_result.fallback
                if gpupixel_result.fallback_reason:
                    frame_summary["gpupixel_fallback_reason"] = gpupixel_result.fallback_reason
            except Exception as exc:
                frame_summary["gpupixel_error"] = str(exc)

        if yolo_seg_aliyun is not None:
            try:
                yt0 = time.perf_counter()
                branch_dir = out_root / "yolo_seg_aliyun" / "cutouts" / group
                branch_result = yolo_seg_aliyun.segment_image_file(
                    capture_path,
                    branch_dir / f"{stem}.png",
                    result,
                )
                branch_image = branch_result.image
                refine_result = refine_subject_edge(branch_image, edge_cfg) if edge_cfg.enabled else None
                if refine_result is not None:
                    save_edge_refine_debug(
                        branch_image,
                        refine_result,
                        output_dir=out_root / "debug",
                        stem=f"{stem}_yolo_seg_aliyun",
                        config=edge_cfg,
                    )
                    branch_image = refine_result.image
                branch_image = apply_instance_alpha_to_roi(
                    branch_image,
                    result,
                    branch_result.roi_box,
                    visitor_mask_dilate_px=yolo_seg_aliyun.config.visitor_mask_dilate_px,
                )
                branch_image.save(branch_dir / f"{stem}.png", format="PNG")
                yolo_seg_aliyun_cutouts.append(branch_image)
                frame_summary["yolo_seg_aliyun_elapsed_seconds"] = time.perf_counter() - yt0
                frame_summary["yolo_seg_aliyun_metrics"] = compute_matting_metrics(
                    branch_image,
                    result,
                    roi_box=branch_result.roi_box,
                )
                frame_summary["yolo_seg_aliyun_preclean"] = {
                    "preclean_pixels": branch_result.preclean_pixels,
                    "preclean_fallback": branch_result.preclean_fallback,
                    "roi_box": branch_result.roi_box,
                }
            except Exception as exc:
                frame_summary["yolo_seg_aliyun_error"] = str(exc)

        if modelscope_universal is not None:
            try:
                mst0 = time.perf_counter()
                branch_dir = out_root / "modelscope_universal_matting" / "cutouts" / group
                branch_dir.mkdir(parents=True, exist_ok=True)
                branch_image = modelscope_universal.segment_image_file(capture_path)
                branch_image = apply_modelscope_universal_constraints(branch_image, result)
                refine_result = refine_subject_edge(branch_image, edge_cfg) if edge_cfg.enabled else None
                if refine_result is not None:
                    save_edge_refine_debug(
                        branch_image,
                        refine_result,
                        output_dir=out_root / "debug",
                        stem=f"{stem}_modelscope_universal",
                        config=edge_cfg,
                    )
                    branch_image = refine_result.image
                    branch_image = apply_modelscope_universal_constraints(branch_image, result)
                branch_image.save(branch_dir / f"{stem}.png", format="PNG")
                modelscope_universal_cutouts.append(branch_image)
                frame_summary["modelscope_universal_elapsed_seconds"] = time.perf_counter() - mst0
                frame_summary["modelscope_universal_metrics"] = compute_matting_metrics(branch_image, result)
            except Exception as exc:
                frame_summary["modelscope_universal_error"] = str(exc)

        if vitmatte is not None:
            try:
                vt0 = time.perf_counter()
                relaxation = build_vitmatte_directional_constraints(result, vitmatte_directional_cfg)
                _save_vitmatte_directional_debug(out_root / "debug", stem, relaxation)
                constraint = vitmatte.matte_image_file(capture_path, relaxation.result)
                vitmatte_image = constraint.image
                refine_result = refine_subject_edge(vitmatte_image, edge_cfg) if edge_cfg.enabled else None
                if refine_result is not None:
                    save_edge_refine_debug(
                        vitmatte_image,
                        refine_result,
                        output_dir=out_root / "debug",
                        stem=f"{stem}_vitmatte",
                        config=edge_cfg,
                    )
                    vitmatte_image = refine_result.image
                vitmatte_image = restore_constrained_alpha_after_refine(vitmatte_image, relaxation.result)
                vitmatte_cutouts.append(vitmatte_image)
                vitmatte_dir = out_root / "yolo_seg_vitmatte" / "cutouts" / group
                vitmatte_dir.mkdir(parents=True, exist_ok=True)
                vitmatte_image.save(vitmatte_dir / f"{stem}.png", format="PNG")
                frame_summary["vitmatte_elapsed_seconds"] = time.perf_counter() - vt0
                (
                    frame_summary["vitmatte_metrics"],
                    frame_summary["vitmatte_original_core_metrics"],
                ) = compute_vitmatte_directional_metrics(vitmatte_image, result, relaxation.result)
                alpha = np.array(vitmatte_image.convert("RGBA").getchannel("A"))
                removed_in_relaxed = int(np.count_nonzero(relaxation.directional_unknown & (alpha <= 16)))
                frame_summary["vitmatte_directional_relax"] = {
                    "side": relaxation.side,
                    "contact_visitors_count": relaxation.contact_visitors_count,
                    "contact_zone_px": int(np.count_nonzero(relaxation.contact_zone)),
                    "relaxed_px": relaxation.relaxed_px,
                    "alpha_removed_in_contact_zone_px": removed_in_relaxed,
                    "sure_background_alpha_px": int(
                        np.count_nonzero(relaxation.result.sure_background & (alpha > 16))
                    ),
                    "original_core_missing_ratio": frame_summary["vitmatte_original_core_metrics"][
                        "subject_core_missing_ratio"
                    ],
                }
                print(
                    f"[ViTMatteDirectional] side={relaxation.side or 'none'} "
                    f"contacts={relaxation.contact_visitors_count} relaxed_px={relaxation.relaxed_px} "
                    f"removed_alpha_px={removed_in_relaxed}"
                )
                frame_summary["vitmatte_constraint"] = {
                    "forced_foreground_px": constraint.forced_foreground_px,
                    "forced_background_px": constraint.forced_background_px,
                    "unknown_px": constraint.unknown_px,
                }
            except Exception as exc:
                frame_summary["vitmatte_error"] = str(exc)

        frame_summary["ok"] = True
        group_summary["frames"].append(frame_summary)

    # MatAnyone branch: process all 4 frames as a video clip
    if matanyone is not None:
        try:
            mt0 = time.perf_counter()
            frames_dir = out_root / "matanyone_frames" / group
            frames_dir.mkdir(parents=True, exist_ok=True)
            frame_paths = [captures_dir / f"{group}_{i}.jpg" for i in range(1, 5)]
            for i, fp in enumerate(frame_paths, start=1):
                if fp.exists():
                    img = Image.open(fp)
                    img.save(frames_dir / f"{group}_{i:04d}.png")

            first_result = frame_results[0]
            if first_result is None:
                raise ValueError("MatAnyone requires an instance result for the first frame.")
            mask_arr = build_matanyone_initial_mask(first_result, matanyone_cfg)
            debug_dir = out_root / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask_arr, "L").save(
                debug_dir / f"{group}_1_matanyone_first_subject_mask.png",
                format="PNG",
            )

            matanyone_branch = (
                "matanyone_bbox_baseline"
                if matanyone_cfg.initial_mask_mode == "bbox"
                else "matanyone_subject_constrained"
            )
            matanyone_dir = out_root / matanyone_branch / "cutouts" / group
            matanyone_dir.mkdir(parents=True, exist_ok=True)

            alpha_frames = matanyone.process_video(
                str(frames_dir),
                mask_arr,
                str(out_root / "matanyone_output" / group),
                save_frames=True,
            )

            for idx, alpha in enumerate(alpha_frames):
                if idx >= 4:
                    continue
                frame_idx = idx
                stem = f"{group}_{frame_idx + 1}"
                frame_result = frame_results[frame_idx]
                target_summary = frame_summaries[frame_idx]
                if frame_result is None or target_summary is None:
                    continue
                source_rgb = Image.open(frame_paths[frame_idx]).convert("RGB")
                initial_constraint = apply_matanyone_alpha_constraints(
                    source_rgb,
                    alpha,
                    frame_result,
                    matanyone_cfg,
                )
                alpha_rgb = initial_constraint.image
                # Apply edge refinement
                refine_result = refine_subject_edge(alpha_rgb, edge_cfg) if edge_cfg.enabled else None
                if refine_result is not None:
                    alpha_rgb = refine_result.image
                final_constraint = apply_matanyone_alpha_constraints(
                    source_rgb,
                    np.array(alpha_rgb.getchannel("A"), dtype=np.uint8),
                    frame_result,
                    matanyone_cfg,
                )
                alpha_rgb = final_constraint.image
                _save_matanyone_debug(out_root / "debug", stem, initial_constraint, refined_image=alpha_rgb)
                alpha_rgb.save(matanyone_dir / f"{stem}.png", format="PNG")
                matanyone_cutouts.append(alpha_rgb)
                frame_metrics = compute_matanyone_metrics(final_constraint, frame_result)
                frame_metrics["raw_foreground_px"] = initial_constraint.raw_foreground_px
                target_summary["matanyone_metrics"] = frame_metrics
                target_summary["matanyone_elapsed_seconds"] = time.perf_counter() - mt0
        except Exception as exc:
            for target_summary in frame_summaries:
                if target_summary is not None:
                    target_summary["matanyone_error"] = str(exc)

    for label, cutouts in (
        ("current_aliyun", current_aliyun_cutouts),
        ("gpupixel_color_beauty_eval", gpupixel_cutouts),
        ("yolo_seg_mask", mask_cutouts),
        ("yolo_seg_modnet", modnet_cutouts),
        ("yolo_seg_aliyun", yolo_seg_aliyun_cutouts),
        ("yolo_seg_vitmatte", vitmatte_cutouts),
        ("modelscope_universal_matting", modelscope_universal_cutouts),
        (
            "matanyone_bbox_baseline" if matanyone_cfg.initial_mask_mode == "bbox" else "matanyone_subject_constrained",
            matanyone_cutouts,
        ),
    ):
        if not cutouts:
            continue
        fused, report = fuse_subjects_temporally(cutouts, temporal_cfg, debug_dir=out_root / "debug", debug_stem=f"{group}_{label}")
        final_items: list[tuple[Image.Image, str]] = []
        cutout_items: list[tuple[Image.Image, str]] = []
        for i, subject in enumerate(fused, start=1):
            if subject is None:
                continue
            bbox = frame_bboxes[i - 1] if i - 1 < len(frame_bboxes) else None
            final = _compose_final(subject, background_item, slogan, slogan_row, subject_bbox=bbox)
            final_dir = out_root / label / "final" / group
            final_dir.mkdir(parents=True, exist_ok=True)
            final.convert("RGB").save(final_dir / f"{group}_{i}.jpg", format="JPEG", quality=92)
            final_items.append((final, f"{label}-{i}"))
            cutout_items.append((subject, f"{label}-{i}"))
        _save_sheet(cutout_items, out_root / f"{group}_{label}_cutout_sheet.jpg")
        _save_sheet(final_items, out_root / f"{group}_{label}_final_sheet.jpg")
        group_summary[f"{label}_fusion"] = {
            "alignment_success_count": report.alignment_success_count,
            "alpha_stable_ratio": report.alpha_stable_ratio,
            "removed_temporal_noise_px": report.removed_temporal_noise_px,
            "fallback_reason": report.fallback_reason,
        }

    generate_comparison_sheet(
        out_root=out_root,
        group=group,
        kind="cutouts",
        output_path=out_root / f"{group}_four_way_cutout_sheet.jpg",
    )
    generate_comparison_sheet(
        out_root=out_root,
        group=group,
        kind="final",
        output_path=out_root / f"{group}_four_way_final_sheet.jpg",
    )
    return group_summary


def run_eval(
    *,
    groups: list[str],
    captures_dir: Path,
    yolo_seg_model_path: str,
    modnet_repo_path: Path,
    modnet_checkpoint_path: Path,
    output_root: Path | None = None,
    instance_config: InstanceSegmentationConfig | None = None,
    sure_fg_erode_px: int | None = None,
    subject_unknown_dilate_px: int | None = None,
    visitor_bg_dilate_px: int | None = None,
    segmenter: SubjectInstanceSegmenter | None = None,
    modnet: ModnetMattingService | None = None,
    modnet_status: str | None = None,
    include_current_aliyun: bool = False,
    current_aliyun: Any | None = None,
    current_aliyun_status: str | None = None,
    include_gpupixel_color_beauty_eval: bool = False,
    gpupixel_preset: str = "light_beauty_color_fix_v1",
    include_yolo_seg_aliyun: bool = False,
    yolo_seg_aliyun: YoloSegAliyunService | None = None,
    yolo_seg_aliyun_status: str | None = None,
    yolo_seg_aliyun_config: YoloSegAliyunConfig | None = None,
    yolo_seg_aliyun_visitor_mask_dilate_px: int | None = None,
    yolo_seg_aliyun_inpaint_radius: int | None = None,
    include_modelscope_universal_matting: bool = False,
    modelscope_universal: ModelScopeUniversalMattingService | None = None,
    modelscope_universal_status: str | None = None,
    modelscope_universal_model_id: str = MODELSCOPE_UNIVERSAL_MODEL_ID,
    include_yolo_seg_vitmatte: bool = False,
    vitmatte: VitmatteService | None = None,
    vitmatte_status: str | None = None,
    vitmatte_directional_relax_enabled: bool = True,
    vitmatte_directional_relax_mode: str = "contact_local",
    vitmatte_contact_side_erode_px: int = 32,
    vitmatte_contact_search_px: int = 24,
    vitmatte_contact_unknown_depth_px: int = 12,
    vitmatte_contact_vertical_margin_px: int = 16,
    vitmatte_contact_side_min_vertical_overlap_ratio: float = 0.15,
    vitmatte_contact_side: str = "auto",
    include_matanyone: bool = False,
    matanyone: MatAnyoneService | None = None,
    matanyone_status: str | None = None,
    matanyone_initial_mask_mode: str = "subject_instance_mask",
    matanyone_core_erode_px: int = 6,
    matanyone_body_soft_band_px: int = 5,
    matanyone_head_soft_band_px: int = 12,
    matanyone_head_height_ratio: float = 0.34,
    matanyone_visitor_clear_dilate_px: int = 8,
    matanyone_hair_side_refine_enabled: bool = True,
    matanyone_hair_refine_side: str = "right",
    matanyone_hair_refine_height_ratio: float = 0.28,
    matanyone_hair_refine_inner_rejudge_px: int = 5,
    matanyone_hair_refine_outer_soft_band_px: int = 4,
    matanyone_hair_refine_min_alpha: int = 16,
    edge_open_kernel_px: int | None = None,
    edge_feather_radius_px: float | None = None,
    edge_effective_bbox_alpha_threshold: int | None = None,
) -> Path:
    APP_STATE["config"] = load_config()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = output_root or (OUTPUT_DIR / "yolo_seg_matting_eval" / timestamp)
    out_root.mkdir(parents=True, exist_ok=True)

    resolved_instance_config = instance_config or build_instance_config(
        yolo_seg_model_path=yolo_seg_model_path,
        sure_fg_erode_px=sure_fg_erode_px,
        subject_unknown_dilate_px=subject_unknown_dilate_px,
        visitor_bg_dilate_px=visitor_bg_dilate_px,
    )
    if segmenter is None:
        segmenter = SubjectInstanceSegmenter(
            resolved_instance_config,
            output_dir=out_root / "debug",
        )
    else:
        segmenter = SubjectInstanceSegmenter(
            resolved_instance_config,
            detector=segmenter._detector,
            output_dir=out_root / "debug",
        )
    edge_cfg = SubjectEdgeRefineConfig.from_mapping(APP_STATE["config"].get("subject_edge_refine", {}))
    edge_overrides: dict[str, Any] = {}
    if edge_open_kernel_px is not None:
        edge_overrides["open_kernel_px"] = int(edge_open_kernel_px)
    if edge_feather_radius_px is not None:
        edge_overrides["feather_radius_px"] = float(edge_feather_radius_px)
    if edge_effective_bbox_alpha_threshold is not None:
        edge_overrides["effective_bbox_alpha_threshold"] = int(edge_effective_bbox_alpha_threshold)
    if edge_overrides:
        edge_cfg = SubjectEdgeRefineConfig(**{**asdict(edge_cfg), **edge_overrides})
    temporal_cfg = TemporalSubjectFusionConfig.from_mapping(APP_STATE["config"].get("temporal_subject_fusion", {}))
    vitmatte_directional_cfg = VitmatteDirectionalRelaxConfig(
        enabled=bool(vitmatte_directional_relax_enabled),
        mode=str(vitmatte_directional_relax_mode),
        contact_side_erode_px=int(vitmatte_contact_side_erode_px),
        contact_search_px=int(vitmatte_contact_search_px),
        contact_unknown_depth_px=int(vitmatte_contact_unknown_depth_px),
        contact_vertical_margin_px=int(vitmatte_contact_vertical_margin_px),
        min_vertical_overlap_ratio=float(vitmatte_contact_side_min_vertical_overlap_ratio),
        contact_side=str(vitmatte_contact_side),
    )
    matanyone_cfg = MatAnyoneConstraintConfig(
        initial_mask_mode=str(matanyone_initial_mask_mode),
        core_erode_px=int(matanyone_core_erode_px),
        body_soft_band_px=int(matanyone_body_soft_band_px),
        head_soft_band_px=int(matanyone_head_soft_band_px),
        head_height_ratio=float(matanyone_head_height_ratio),
        visitor_clear_dilate_px=int(matanyone_visitor_clear_dilate_px),
        hair_side_refine_enabled=bool(matanyone_hair_side_refine_enabled),
        hair_refine_side=str(matanyone_hair_refine_side),
        hair_refine_height_ratio=float(matanyone_hair_refine_height_ratio),
        hair_refine_inner_rejudge_px=int(matanyone_hair_refine_inner_rejudge_px),
        hair_refine_outer_soft_band_px=int(matanyone_hair_refine_outer_soft_band_px),
        hair_refine_min_alpha=int(matanyone_hair_refine_min_alpha),
    )
    snapshot = get_rotation_snapshot()
    slogan = snapshot.get("slogan_content") or snapshot["slogan"]
    slogan_row = int(snapshot.get("slogan_row", 1))
    background_item = get_background_items()[0]

    if modnet_status is not None:
        matting_status = modnet_status
    elif modnet is not None:
        matting_status = "available"
    else:
        matting_status = "available"
        try:
            modnet = ModnetMattingService(
                repo_path=modnet_repo_path,
                checkpoint_path=modnet_checkpoint_path,
                prefer_cuda=True,
            )
        except Exception as exc:
            modnet = None
            matting_status = f"matting_not_available: {exc}"
            print(f"[YOLO-SEG-MATTING] {matting_status}")

    require_current_aliyun = include_current_aliyun or include_gpupixel_color_beauty_eval
    if current_aliyun_status is not None:
        aliyun_status = current_aliyun_status
    elif current_aliyun is not None:
        aliyun_status = "available"
    elif require_current_aliyun:
        aliyun_status = "available"
        try:
            current_aliyun = AliSegmentService(APP_STATE["config"])
        except Exception as exc:
            current_aliyun = None
            aliyun_status = f"current_aliyun_not_available: {exc}"
            print(f"[YOLO-SEG-MATTING] {aliyun_status}")
    else:
        aliyun_status = "not_requested"

    if yolo_seg_aliyun_status is not None:
        seg_aliyun_status = yolo_seg_aliyun_status
    elif yolo_seg_aliyun is not None:
        seg_aliyun_status = "available"
    elif include_yolo_seg_aliyun:
        seg_aliyun_status = "available"
        try:
            resolved_seg_aliyun_config = yolo_seg_aliyun_config or YoloSegAliyunConfig()
            seg_aliyun_overrides: dict[str, Any] = {}
            if yolo_seg_aliyun_visitor_mask_dilate_px is not None:
                seg_aliyun_overrides["visitor_mask_dilate_px"] = int(yolo_seg_aliyun_visitor_mask_dilate_px)
            if yolo_seg_aliyun_inpaint_radius is not None:
                seg_aliyun_overrides["inpaint_radius"] = int(yolo_seg_aliyun_inpaint_radius)
            if seg_aliyun_overrides:
                resolved_seg_aliyun_config = YoloSegAliyunConfig(
                    **{**asdict(resolved_seg_aliyun_config), **seg_aliyun_overrides}
                )
            yolo_seg_aliyun = YoloSegAliyunService(
                _create_pipeline(APP_STATE["config"].get("matting_api", {})),
                resolved_seg_aliyun_config,
                debug_dir=out_root / "debug",
            )
        except Exception as exc:
            yolo_seg_aliyun = None
            seg_aliyun_status = f"yolo_seg_aliyun_not_available: {exc}"
            print(f"[YOLO-SEG-MATTING] {seg_aliyun_status}")
    else:
        seg_aliyun_status = "not_requested"

    if modelscope_universal_status is not None:
        modelscope_status = modelscope_universal_status
    elif modelscope_universal is not None:
        modelscope_status = "available"
    elif include_modelscope_universal_matting:
        modelscope_status = "available"
        try:
            modelscope_universal = ModelScopeUniversalMattingService(model_id=modelscope_universal_model_id)
            modelscope_universal.warmup()
        except Exception as exc:
            modelscope_universal = None
            modelscope_status = f"modelscope_universal_not_available: {exc}"
            print(f"[YOLO-SEG-MATTING] {modelscope_status}")
    else:
        modelscope_status = "not_requested"

    if vitmatte_status is not None:
        vitmatte_eval_status = vitmatte_status
    elif vitmatte is not None:
        vitmatte_eval_status = "available"
    elif include_yolo_seg_vitmatte:
        vitmatte_eval_status = "available"
        try:
            vitmatte = VitmatteService(prefer_cuda=True)
        except Exception as exc:
            vitmatte = None
            vitmatte_eval_status = f"vitmatte_not_available: {exc}"
            print(f"[YOLO-SEG-MATTING] {vitmatte_eval_status}")
    else:
        vitmatte_eval_status = "not_requested"

    if matanyone_status is not None:
        matanyone_eval_status = matanyone_status
    elif matanyone is not None:
        matanyone_eval_status = "available"
    elif include_matanyone:
        matanyone_eval_status = "available"
        try:
            matanyone = MatAnyoneService(prefer_cuda=True)
        except Exception as exc:
            matanyone = None
            matanyone_eval_status = f"matanyone_not_available: {exc}"
            print(f"[YOLO-SEG-MATTING] {matanyone_eval_status}")
    else:
        matanyone_eval_status = "not_requested"

    summary: dict[str, Any] = {
        "generated_at": timestamp,
        "yolo_seg_model_path": yolo_seg_model_path,
        "modnet_repo_path": str(modnet_repo_path),
        "modnet_checkpoint_path": str(modnet_checkpoint_path),
        "matting_status": matting_status,
        "current_aliyun_status": aliyun_status,
        "gpupixel_color_beauty_eval_status": (
            "available"
            if include_gpupixel_color_beauty_eval and current_aliyun is not None
            else ("requires_current_aliyun" if include_gpupixel_color_beauty_eval else "not_requested")
        ),
        "gpupixel_preset": gpupixel_preset,
        "yolo_seg_aliyun_status": seg_aliyun_status,
        "modelscope_universal_status": modelscope_status,
        "modelscope_universal_model_id": modelscope_universal_model_id,
        "vitmatte_status": vitmatte_eval_status,
        "vitmatte_directional_relax_config": asdict(vitmatte_directional_cfg),
        "matanyone_status": matanyone_eval_status,
        "matanyone_constraint_config": asdict(matanyone_cfg),
        "trimap_config": trimap_config_to_dict(resolved_instance_config),
        "groups": {},
    }
    for group in groups:
        summary["groups"][group] = _process_group(
            group,
            captures_dir,
            out_root,
            segmenter,
            modnet,
            current_aliyun,
            include_gpupixel_color_beauty_eval,
            gpupixel_preset,
            yolo_seg_aliyun,
            modelscope_universal,
            vitmatte,
            vitmatte_directional_cfg,
            matanyone,
            matanyone_cfg,
            edge_cfg,
            temporal_cfg,
            background_item,
            slogan,
            slogan_row,
        )

    summary["aggregate_metrics"] = aggregate_comparison_metrics(summary)
    summary_path = out_root / "summary_metrics.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[YOLO-SEG-MATTING] done: {summary_path}")
    return out_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline YOLO-seg + MODNet constrained matting evaluation.")
    parser.add_argument("--groups", nargs="+", default=DEFAULT_GROUPS)
    parser.add_argument("--captures-dir", default=str(BASE_DIR / "generated" / "captures"))
    parser.add_argument("--yolo-seg-model-path", default=str(BASE_DIR / "models" / "yolo11x-seg.pt"))
    parser.add_argument("--modnet-repo-path", default=str(BASE_DIR / "MODNet"))
    parser.add_argument("--modnet-checkpoint-path", default=str(BASE_DIR / "models" / "modnet_photographic_portrait_matting.ckpt"))
    parser.add_argument("--sure-fg-erode-px", type=int, default=None)
    parser.add_argument("--subject-unknown-dilate-px", type=int, default=None)
    parser.add_argument("--visitor-bg-dilate-px", type=int, default=None)
    parser.add_argument("--include-current-aliyun", action="store_true")
    parser.add_argument("--include-gpupixel-color-beauty-eval", action="store_true")
    parser.add_argument("--gpupixel-preset", default="light_beauty_color_fix_v1")
    parser.add_argument("--include-yolo-seg-aliyun", action="store_true")
    parser.add_argument("--include-modelscope-universal-matting", action="store_true")
    parser.add_argument("--modelscope-universal-model-id", default=MODELSCOPE_UNIVERSAL_MODEL_ID)
    parser.add_argument("--include-yolo-seg-vitmatte", action="store_true")
    parser.add_argument("--include-matanyone", action="store_true")
    parser.add_argument("--matanyone-initial-mask-mode", choices=("subject_instance_mask", "bbox"), default="subject_instance_mask")
    parser.add_argument("--matanyone-core-erode-px", type=int, default=6)
    parser.add_argument("--matanyone-body-soft-band-px", type=int, default=5)
    parser.add_argument("--matanyone-head-soft-band-px", type=int, default=12)
    parser.add_argument("--matanyone-head-height-ratio", type=float, default=0.34)
    parser.add_argument("--matanyone-visitor-clear-dilate-px", type=int, default=8)
    parser.set_defaults(matanyone_hair_side_refine_enabled=True)
    parser.add_argument("--matanyone-hair-side-refine-enabled", dest="matanyone_hair_side_refine_enabled", action="store_true")
    parser.add_argument("--disable-matanyone-hair-side-refine", dest="matanyone_hair_side_refine_enabled", action="store_false")
    parser.add_argument("--matanyone-hair-refine-side", choices=("left", "right"), default="right")
    parser.add_argument("--matanyone-hair-refine-height-ratio", type=float, default=0.28)
    parser.add_argument("--matanyone-hair-refine-inner-rejudge-px", type=int, default=5)
    parser.add_argument("--matanyone-hair-refine-outer-soft-band-px", type=int, default=4)
    parser.add_argument("--matanyone-hair-refine-min-alpha", type=int, default=16)
    parser.add_argument("--disable-vitmatte-directional-relax", action="store_true")
    parser.add_argument("--vitmatte-directional-relax-mode", choices=("contact_local", "side_band"), default="contact_local")
    parser.add_argument("--vitmatte-contact-side-erode-px", type=int, default=32)
    parser.add_argument("--vitmatte-contact-search-px", type=int, default=24)
    parser.add_argument("--vitmatte-contact-unknown-depth-px", type=int, default=12)
    parser.add_argument("--vitmatte-contact-vertical-margin-px", type=int, default=16)
    parser.add_argument("--vitmatte-contact-side-min-vertical-overlap-ratio", type=float, default=0.15)
    parser.add_argument("--vitmatte-contact-side", choices=("auto", "left", "right"), default="auto")
    parser.add_argument("--yolo-seg-aliyun-visitor-mask-dilate-px", type=int, default=None)
    parser.add_argument("--yolo-seg-aliyun-inpaint-radius", type=int, default=None)
    parser.add_argument("--edge-open-kernel-px", type=int, default=None)
    parser.add_argument("--edge-feather-radius-px", type=float, default=None)
    parser.add_argument("--edge-effective-bbox-alpha-threshold", type=int, default=None)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    run_eval(
        groups=args.groups,
        captures_dir=Path(args.captures_dir),
        yolo_seg_model_path=args.yolo_seg_model_path,
        modnet_repo_path=Path(args.modnet_repo_path),
        modnet_checkpoint_path=Path(args.modnet_checkpoint_path),
        sure_fg_erode_px=args.sure_fg_erode_px,
        subject_unknown_dilate_px=args.subject_unknown_dilate_px,
        visitor_bg_dilate_px=args.visitor_bg_dilate_px,
        include_current_aliyun=args.include_current_aliyun,
        include_gpupixel_color_beauty_eval=args.include_gpupixel_color_beauty_eval,
        gpupixel_preset=args.gpupixel_preset,
        include_yolo_seg_aliyun=args.include_yolo_seg_aliyun,
        include_modelscope_universal_matting=args.include_modelscope_universal_matting,
        modelscope_universal_model_id=args.modelscope_universal_model_id,
        include_yolo_seg_vitmatte=args.include_yolo_seg_vitmatte,
        vitmatte_directional_relax_enabled=not args.disable_vitmatte_directional_relax,
        vitmatte_directional_relax_mode=args.vitmatte_directional_relax_mode,
        vitmatte_contact_side_erode_px=args.vitmatte_contact_side_erode_px,
        vitmatte_contact_search_px=args.vitmatte_contact_search_px,
        vitmatte_contact_unknown_depth_px=args.vitmatte_contact_unknown_depth_px,
        vitmatte_contact_vertical_margin_px=args.vitmatte_contact_vertical_margin_px,
        vitmatte_contact_side_min_vertical_overlap_ratio=args.vitmatte_contact_side_min_vertical_overlap_ratio,
        vitmatte_contact_side=args.vitmatte_contact_side,
        include_matanyone=args.include_matanyone,
        matanyone_initial_mask_mode=args.matanyone_initial_mask_mode,
        matanyone_core_erode_px=args.matanyone_core_erode_px,
        matanyone_body_soft_band_px=args.matanyone_body_soft_band_px,
        matanyone_head_soft_band_px=args.matanyone_head_soft_band_px,
        matanyone_head_height_ratio=args.matanyone_head_height_ratio,
        matanyone_visitor_clear_dilate_px=args.matanyone_visitor_clear_dilate_px,
        matanyone_hair_side_refine_enabled=args.matanyone_hair_side_refine_enabled,
        matanyone_hair_refine_side=args.matanyone_hair_refine_side,
        matanyone_hair_refine_height_ratio=args.matanyone_hair_refine_height_ratio,
        matanyone_hair_refine_inner_rejudge_px=args.matanyone_hair_refine_inner_rejudge_px,
        matanyone_hair_refine_outer_soft_band_px=args.matanyone_hair_refine_outer_soft_band_px,
        matanyone_hair_refine_min_alpha=args.matanyone_hair_refine_min_alpha,
        yolo_seg_aliyun_visitor_mask_dilate_px=args.yolo_seg_aliyun_visitor_mask_dilate_px,
        yolo_seg_aliyun_inpaint_radius=args.yolo_seg_aliyun_inpaint_radius,
        edge_open_kernel_px=args.edge_open_kernel_px,
        edge_feather_radius_px=args.edge_feather_radius_px,
        edge_effective_bbox_alpha_threshold=args.edge_effective_bbox_alpha_threshold,
        output_root=Path(args.output_root) if args.output_root else None,
    )


if __name__ == "__main__":
    main()
