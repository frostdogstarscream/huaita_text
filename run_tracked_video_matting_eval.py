from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from matanyone_service import (
    MatAnyoneConstraintConfig,
    MatAnyoneConstraintMasks,
    MatAnyoneConstraintResult,
    apply_matanyone_alpha_constraints,
)
from subject_edge_refine import SubjectEdgeRefineConfig, refine_subject_edge
from subject_instance_segmentation import InstanceCandidate, InstanceSegmentationConfig, InstanceSegmentationResult
from subject_instance_tracking import SubjectInstanceTracker, TrackingConfig, TrackedFrameResult


def select_output_frames(frames: list[Any], output_indices: list[int]) -> list[Any]:
    if len(output_indices) != 4 or any(index < 0 or index >= len(frames) for index in output_indices):
        raise ValueError("Exactly four valid output frame indices are required.")
    return [frames[index] for index in output_indices]


def apply_tracked_alpha_constraints(
    image: Image.Image,
    raw_alpha: np.ndarray,
    frame: TrackedFrameResult,
) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = np.asarray(raw_alpha, dtype=np.uint8).copy()
    expected_shape = (rgba.height, rgba.width)
    if alpha.shape != expected_shape:
        raise ValueError(f"raw_alpha shape {alpha.shape} does not match image size {expected_shape}")
    alpha[frame.sure_background.astype(bool)] = 0
    alpha[frame.sure_foreground.astype(bool)] = 255
    pixels = np.array(rgba)
    pixels[:, :, 3] = alpha
    return Image.fromarray(pixels, "RGBA")


def _combined_visitor_mask(frame: TrackedFrameResult) -> np.ndarray:
    mask = np.zeros(frame.sure_foreground.shape, dtype=bool)
    for visitor in frame.visitors:
        mask |= visitor.mask.astype(bool)
    return mask


def compute_tracked_metrics(image: Image.Image, frame: TrackedFrameResult) -> dict[str, float]:
    alpha = np.asarray(image.convert("RGBA").getchannel("A"), dtype=np.uint8)
    visitor_mask = _combined_visitor_mask(frame)
    subject_mask = frame.selected.mask.astype(bool) if frame.selected is not None else np.zeros(alpha.shape, dtype=bool)
    core_mask = frame.sure_foreground.astype(bool)
    visitor_area = int(np.count_nonzero(visitor_mask))
    core_area = int(np.count_nonzero(core_mask))
    foreground_sum = float(alpha.sum())
    outside_sum = float(alpha[~subject_mask].sum()) if np.any(~subject_mask) else 0.0
    return {
        "visitor_track_alpha_ratio": (
            float(alpha[visitor_mask].sum()) / float(visitor_area * 255) if visitor_area else 0.0
        ),
        "subject_core_missing_ratio": (
            float(np.count_nonzero(alpha[core_mask] < 16)) / float(core_area) if core_area else 0.0
        ),
        "outside_subject_soft_alpha_ratio": outside_sum / foreground_sum if foreground_sum else 0.0,
        "foreground_px": float(np.count_nonzero(alpha > 8)),
    }


def _bbox_center(frame: TrackedFrameResult) -> tuple[float, float]:
    if frame.selected is None:
        return (0.0, 0.0)
    left, top, right, bottom = frame.selected.bbox
    return ((left + right) / 2.0, (top + bottom) / 2.0)


def compute_edge_temporal_jitter(
    images: list[Image.Image],
    frames: list[TrackedFrameResult],
) -> float:
    if len(images) != len(frames):
        raise ValueError("images and frames must have identical lengths.")
    if len(images) < 2:
        return 0.0
    jitter_scores: list[float] = []
    previous_alpha = np.asarray(images[0].convert("RGBA").getchannel("A"), dtype=np.uint8)
    previous_center = _bbox_center(frames[0])
    previous_edge = cv2.Canny(previous_alpha, 32, 96) > 0
    for image, frame in zip(images[1:], frames[1:]):
        current_alpha = np.asarray(image.convert("RGBA").getchannel("A"), dtype=np.uint8)
        current_center = _bbox_center(frame)
        transform = np.float32(
            [
                [1, 0, previous_center[0] - current_center[0]],
                [0, 1, previous_center[1] - current_center[1]],
            ]
        )
        aligned_alpha = cv2.warpAffine(
            current_alpha,
            transform,
            (current_alpha.shape[1], current_alpha.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )
        current_edge = cv2.Canny(aligned_alpha, 32, 96) > 0
        union = int(np.count_nonzero(previous_edge | current_edge))
        xor = int(np.count_nonzero(previous_edge ^ current_edge))
        jitter_scores.append(float(xor) / float(union) if union else 0.0)
        previous_alpha = aligned_alpha
        previous_center = previous_center
        previous_edge = current_edge
    return float(np.mean(jitter_scores)) if jitter_scores else 0.0


def aggregate_metric_rows(
    rows: list[dict[str, float]],
    *,
    elapsed_seconds: float,
) -> dict[str, float]:
    summary: dict[str, float] = {"elapsed_seconds": float(elapsed_seconds)}
    if not rows:
        return summary
    for key in rows[0]:
        values = [float(row[key]) for row in rows if key in row]
        if values:
            summary[f"{key}_avg"] = float(np.mean(values))
            summary[f"{key}_max"] = float(np.max(values))
    return summary


def tracked_frame_to_instance_result(
    frame: TrackedFrameResult,
    *,
    source_path: Path,
    image_size: tuple[int, int],
) -> InstanceSegmentationResult:
    if frame.selected is None:
        raise ValueError("Tracked frame has no selected subject.")
    selected = InstanceCandidate(
        bbox=frame.selected.bbox,
        confidence=frame.selected.confidence,
        mask=frame.selected.mask,
        score=frame.selected.score,
    )
    visitors = [
        InstanceCandidate(
            bbox=visitor.bbox,
            confidence=visitor.confidence,
            mask=visitor.mask,
            score=visitor.score,
        )
        for visitor in frame.visitors
    ]
    return InstanceSegmentationResult(
        source_path=source_path,
        image_size=image_size,
        selected=selected,
        candidates=[selected, *visitors],
        visitors=visitors,
        trimap=frame.trimap,
        sure_foreground=frame.sure_foreground,
        sure_background=frame.sure_background,
        unknown=frame.unknown,
    )


def _refine_and_reconstrain(
    cutout: Image.Image,
    frame: TrackedFrameResult,
    edge_config: SubjectEdgeRefineConfig,
) -> Image.Image:
    refined = refine_subject_edge(cutout, edge_config).image
    return apply_tracked_alpha_constraints(
        refined.convert("RGB"),
        np.asarray(refined.getchannel("A"), dtype=np.uint8),
        frame,
    )


def _save_sheet(items: list[tuple[Image.Image, str]], output_path: Path) -> None:
    if not items:
        return
    cell_width, cell_height, label_height = 240, 360, 30
    canvas = Image.new("RGB", (cell_width * len(items), cell_height + label_height), (244, 241, 232))
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(items):
        preview = image.convert("RGB").copy()
        preview.thumbnail((cell_width - 8, cell_height - 8), Image.Resampling.LANCZOS)
        x = index * cell_width + (cell_width - preview.width) // 2
        y = label_height + (cell_height - preview.height) // 2
        canvas.paste(preview, (x, y))
        draw.text((index * cell_width + 6, 8), label, fill=(35, 35, 35))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="JPEG", quality=92)


def _save_sheet_grid(rows: list[list[tuple[Image.Image, str]]], output_path: Path) -> None:
    rows = [row for row in rows if row]
    if not rows:
        return
    cell_width, cell_height, label_height = 240, 360, 30
    column_count = max(len(row) for row in rows)
    canvas = Image.new(
        "RGB",
        (cell_width * column_count, (cell_height + label_height) * len(rows)),
        (244, 241, 232),
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, row in enumerate(rows):
        row_y = row_index * (cell_height + label_height)
        for column_index, (image, label) in enumerate(row):
            preview = image.convert("RGB").copy()
            preview.thumbnail((cell_width - 8, cell_height - 8), Image.Resampling.LANCZOS)
            x = column_index * cell_width + (cell_width - preview.width) // 2
            y = row_y + label_height + (cell_height - preview.height) // 2
            canvas.paste(preview, (x, y))
            draw.text((column_index * cell_width + 6, row_y + 8), label, fill=(35, 35, 35))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="JPEG", quality=92)


def _make_default_composer() -> Any:
    from app_state import APP_STATE
    from background_manager import get_background_items
    from config_manager import load_config
    from run_yolo_seg_matting_eval import _compose_final
    from slogan_manager import get_rotation_snapshot

    if "config" not in APP_STATE or not APP_STATE.get("config"):
        APP_STATE["config"] = load_config()
    backgrounds = get_background_items()
    snapshot = get_rotation_snapshot()
    slogan = snapshot.get("slogan_content") or snapshot["slogan"]
    slogan_row = int(snapshot.get("slogan_row", 1))

    def compose(subject: Image.Image, _branch: str, output_slot: int) -> Image.Image:
        background = backgrounds[output_slot % len(backgrounds)]
        return _compose_final(subject, background, slogan, slogan_row)

    return compose


def _save_branch_outputs(
    branch: str,
    images: list[Image.Image],
    frames: list[TrackedFrameResult],
    output_dir: Path,
    compose_final: Any,
    extra_metrics: list[dict[str, float]] | None = None,
) -> dict[str, Any]:
    cutout_dir = output_dir / branch / "cutouts"
    final_dir = output_dir / branch / "final"
    cutout_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    cutout_sheet_items = []
    final_sheet_items = []
    for slot, (image, frame) in enumerate(zip(images, frames), start=1):
        cutout_path = cutout_dir / f"{slot:02d}.png"
        image.convert("RGBA").save(cutout_path, format="PNG")
        final = compose_final(image, branch, slot - 1)
        final.convert("RGB").save(final_dir / f"{slot:02d}.jpg", format="JPEG", quality=92)
        metric_row = compute_tracked_metrics(image, frame)
        if extra_metrics is not None:
            metric_row.update(extra_metrics[slot - 1])
        metrics.append(metric_row)
        cutout_sheet_items.append((image, f"{branch}-{slot}"))
        final_sheet_items.append((final, f"{branch}-{slot}"))
    _save_sheet(cutout_sheet_items, output_dir / "sheets" / f"{branch}_cutout_sheet.jpg")
    _save_sheet(final_sheet_items, output_dir / "sheets" / f"{branch}_final_sheet.jpg")
    return {
        "status": "ok",
        "frame_metrics": metrics,
        "aggregate_metrics": aggregate_metric_rows(metrics, elapsed_seconds=0.0),
        "final_count": len(images),
        "_cutout_sheet_items": cutout_sheet_items,
        "_final_sheet_items": final_sheet_items,
    }


def _subject_priority_config() -> MatAnyoneConstraintConfig:
    return MatAnyoneConstraintConfig(
        occlusion_conflict_policy="selected_subject_priority",
        contact_subject_priority_enabled=True,
        contact_visitor_dilate_px=8,
        contact_core_erode_px=6,
        contact_edge_alpha_floor=160,
    )


def _body_refine_config() -> MatAnyoneConstraintConfig:
    return MatAnyoneConstraintConfig(
        occlusion_conflict_policy="selected_subject_priority",
        contact_subject_priority_enabled=True,
        contact_visitor_dilate_px=8,
        contact_core_erode_px=6,
        contact_edge_alpha_floor=160,
        torso_refine_enabled=True,
        torso_inner_rejudge_px=4,
        torso_outer_soft_band_px=1,
        arm_refine_enabled=True,
        arm_inner_rejudge_px=3,
        arm_outer_soft_band_px=1,
        body_contact_edge_alpha_floor=190,
    )


def _subject_priority_metrics(constraint: MatAnyoneConstraintResult) -> dict[str, float]:
    return {
        "contact_conflict_px": float(constraint.contact_conflict_px),
        "contact_core_restored_px": float(constraint.contact_core_restored_px),
        "contact_edge_floor_applied_px": float(constraint.contact_edge_floor_applied_px),
        "visitor_visible_residual_ratio": float(constraint.visitor_visible_residual_ratio),
        "subject_contact_missing_ratio": float(constraint.subject_contact_missing_ratio),
    }


def _body_refine_metrics(constraint: MatAnyoneConstraintResult) -> dict[str, float]:
    return {
        "contact_conflict_px": float(constraint.contact_conflict_px),
        "contact_core_restored_px": float(constraint.contact_core_restored_px),
        "contact_edge_floor_applied_px": float(constraint.contact_edge_floor_applied_px),
        "visitor_visible_residual_ratio": float(constraint.visitor_visible_residual_ratio),
        "subject_contact_missing_ratio": float(constraint.subject_contact_missing_ratio),
        "body_outside_soft_alpha_ratio": float(constraint.body_outside_soft_alpha_ratio),
        "body_edge_removed_px": float(constraint.body_edge_removed_px),
        "body_core_missing_ratio": float(constraint.body_core_missing_ratio),
    }


def _measure_contact_metrics(
    image: Image.Image,
    masks: MatAnyoneConstraintMasks,
    *,
    edge_alpha_floor: int,
) -> dict[str, float]:
    alpha = np.asarray(image.convert("RGBA").getchannel("A"), dtype=np.uint8)
    contact_px = int(np.count_nonzero(masks.contact_conflict))
    visible_visitor_px = int(np.count_nonzero(masks.visitor_visible_clear))
    return {
        "contact_conflict_px": float(contact_px),
        "visitor_visible_residual_ratio": (
            float(np.count_nonzero((alpha > 16) & masks.visitor_visible_clear))
            / float(visible_visitor_px)
            if visible_visitor_px
            else 0.0
        ),
        "subject_contact_missing_ratio": (
            float(np.count_nonzero((alpha < int(edge_alpha_floor)) & masks.contact_conflict))
            / float(contact_px)
            if contact_px
            else 0.0
        ),
    }


def _save_subject_priority_debug(
    constraint: MatAnyoneConstraintResult,
    *,
    output_dir: Path,
    stem: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    masks = constraint.masks
    images = {
        "contact_conflict": masks.contact_conflict,
        "visitor_visible_clear": masks.visitor_visible_clear,
        "contact_core_restore": masks.contact_core,
        "contact_edge_floor": masks.contact_edge,
    }
    for suffix, mask in images.items():
        Image.fromarray(mask.astype(np.uint8) * 255, "L").save(
            output_dir / f"{stem}_{suffix}.png", format="PNG"
        )
    constraint.image.convert("RGBA").save(
        output_dir / f"{stem}_subject_protected_after.png", format="PNG"
    )


def _save_body_refine_debug(
    constraint: MatAnyoneConstraintResult,
    *,
    output_dir: Path,
    stem: str,
) -> None:
    _save_subject_priority_debug(constraint, output_dir=output_dir, stem=stem)
    masks = constraint.masks
    images = {
        "torso_region": masks.torso_region,
        "arm_region": masks.arm_region,
        "body_inner_rejudge": masks.body_inner_rejudge,
        "body_outer_support": masks.body_outer_support,
    }
    for suffix, mask in images.items():
        Image.fromarray(mask.astype(np.uint8) * 255, "L").save(
            output_dir / f"{stem}_{suffix}.png", format="PNG"
        )


def run_take(
    manifest_path: Path,
    output_dir: Path,
    *,
    tracker: Any,
    matanyone: Any | None,
    modnet: Any | None,
    compose_final: Any | None = None,
    edge_config: SubjectEdgeRefineConfig | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_compose_final = compose_final or _make_default_composer()
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    frame_paths = [Path(path) for path in manifest.get("frame_paths", [])]
    output_indices = [int(index) for index in manifest.get("output_frame_indices", [])]
    if manifest.get("status") != "ok":
        return {"status": "input_failed", "error": manifest.get("error"), "branches": {}}
    output_paths = select_output_frames(frame_paths, output_indices)
    sequence = tracker.track_paths(frame_paths)
    if sequence.status != "ok":
        summary = {"status": sequence.status, "error": sequence.error, "branches": {}}
        (output_dir / "summary_metrics.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary
    output_frames = select_output_frames(sequence.frames, output_indices)
    edge_cfg = edge_config or SubjectEdgeRefineConfig()
    branches: dict[str, Any] = {}
    cutout_sheet_rows: list[list[tuple[Image.Image, str]]] = []
    final_sheet_rows: list[list[tuple[Image.Image, str]]] = []

    mask_images: list[Image.Image] = []
    for source_path, frame in zip(output_paths, output_frames):
        with Image.open(source_path) as source:
            raw_alpha = frame.selected.mask.astype(np.uint8) * 255  # type: ignore[union-attr]
            constrained = apply_tracked_alpha_constraints(source.convert("RGB"), raw_alpha, frame)
        mask_images.append(_refine_and_reconstrain(constrained, frame, edge_cfg))
    mask_summary = _save_branch_outputs(
        "tracked_yolo_seg_mask", mask_images, output_frames, output_dir, resolved_compose_final
    )
    cutout_sheet_rows.append(mask_summary.pop("_cutout_sheet_items"))
    final_sheet_rows.append(mask_summary.pop("_final_sheet_items"))
    branches["tracked_yolo_seg_mask"] = mask_summary

    if modnet is not None:
        modnet_started = time.perf_counter()
        try:
            modnet_images = []
            for source_path, frame in zip(output_paths, output_frames):
                with Image.open(source_path) as source:
                    instance = tracked_frame_to_instance_result(frame, source_path=source_path, image_size=source.size)
                raw = modnet.matte_image_file(source_path, instance).image
                constrained = apply_tracked_alpha_constraints(
                    raw.convert("RGB"), np.asarray(raw.getchannel("A"), dtype=np.uint8), frame
                )
                modnet_images.append(_refine_and_reconstrain(constrained, frame, edge_cfg))
            modnet_summary = _save_branch_outputs(
                "tracked_modnet_4frame", modnet_images, output_frames, output_dir, resolved_compose_final
            )
            cutout_sheet_rows.append(modnet_summary.pop("_cutout_sheet_items"))
            final_sheet_rows.append(modnet_summary.pop("_final_sheet_items"))
            modnet_summary["elapsed_seconds"] = time.perf_counter() - modnet_started
            branches["tracked_modnet_4frame"] = modnet_summary
        except Exception as exc:
            branches["tracked_modnet_4frame"] = {
                "status": "failed",
                "error": str(exc),
                "elapsed_seconds": time.perf_counter() - modnet_started,
            }

    if matanyone is not None:
        matanyone_started = time.perf_counter()
        video_path = manifest.get("video_path")
        if not video_path:
            branches["tracked_matanyone"] = {"status": "missing_video"}
        else:
            try:
                alpha_frames = matanyone.process_video(
                    Path(video_path),
                    sequence.frames[0].selected.mask.astype(np.uint8) * 255,  # type: ignore[union-attr]
                    output_dir / "tracked_matanyone" / "raw",
                    save_frames=True,
                )
                selected_alpha = select_output_frames(alpha_frames, output_indices)
                current_images = []
                for source_path, raw_alpha, frame in zip(output_paths, selected_alpha, output_frames):
                    with Image.open(source_path) as source:
                        constrained = apply_tracked_alpha_constraints(
                            source.convert("RGB"), np.asarray(raw_alpha, dtype=np.uint8), frame
                        )
                    current_images.append(_refine_and_reconstrain(constrained, frame, edge_cfg))
                current_summary = _save_branch_outputs(
                    "tracked_matanyone_current", current_images, output_frames, output_dir, resolved_compose_final
                )
                cutout_sheet_rows.append(current_summary.pop("_cutout_sheet_items"))
                final_sheet_rows.append(current_summary.pop("_final_sheet_items"))
                current_summary["edge_temporal_jitter"] = compute_edge_temporal_jitter(
                    current_images, output_frames
                )
                current_summary["elapsed_seconds"] = time.perf_counter() - matanyone_started
                branches["tracked_matanyone_current"] = current_summary

                try:
                    priority_cfg = _subject_priority_config()
                    priority_images: list[Image.Image] = []
                    priority_metrics: list[dict[str, float]] = []
                    current_contact_metrics: list[dict[str, float]] = []
                    debug_dir = output_dir / "tracked_matanyone_subject_priority" / "debug"
                    for slot, (source_path, raw_alpha, frame) in enumerate(
                        zip(output_paths, selected_alpha, output_frames),
                        start=1,
                    ):
                        with Image.open(source_path) as source:
                            instance = tracked_frame_to_instance_result(frame, source_path=source_path, image_size=source.size)
                            priority = apply_matanyone_alpha_constraints(
                                source.convert("RGB"),
                                np.asarray(raw_alpha, dtype=np.uint8),
                                instance,
                                priority_cfg,
                            )
                        refined = refine_subject_edge(priority.image, edge_cfg).image
                        restored = apply_matanyone_alpha_constraints(
                            refined.convert("RGB"),
                            np.asarray(refined.getchannel("A"), dtype=np.uint8),
                            instance,
                            priority_cfg,
                        )
                        priority_images.append(restored.image)
                        priority_metrics.append(_subject_priority_metrics(restored))
                        current_contact_metrics.append(
                            _measure_contact_metrics(
                                current_images[slot - 1],
                                restored.masks,
                                edge_alpha_floor=priority_cfg.contact_edge_alpha_floor,
                            )
                        )
                        _save_subject_priority_debug(restored, output_dir=debug_dir, stem=f"{slot:02d}")
                    for row, contact_metrics in zip(current_summary["frame_metrics"], current_contact_metrics):
                        row.update(contact_metrics)
                    current_summary["aggregate_metrics"] = aggregate_metric_rows(
                        current_summary["frame_metrics"], elapsed_seconds=0.0
                    )
                    priority_summary = _save_branch_outputs(
                        "tracked_matanyone_subject_priority",
                        priority_images,
                        output_frames,
                        output_dir,
                        resolved_compose_final,
                        extra_metrics=priority_metrics,
                    )
                    cutout_sheet_rows.append(priority_summary.pop("_cutout_sheet_items"))
                    final_sheet_rows.append(priority_summary.pop("_final_sheet_items"))
                    priority_summary["edge_temporal_jitter"] = compute_edge_temporal_jitter(
                        priority_images, output_frames
                    )
                    priority_summary["elapsed_seconds"] = time.perf_counter() - matanyone_started
                    branches["tracked_matanyone_subject_priority"] = priority_summary

                except Exception as exc:
                    branches["tracked_matanyone_subject_priority"] = {
                        "status": "failed",
                        "error": str(exc),
                        "elapsed_seconds": time.perf_counter() - matanyone_started,
                    }
                try:
                    body_cfg = _body_refine_config()
                    body_images: list[Image.Image] = []
                    body_metrics: list[dict[str, float]] = []
                    debug_dir = output_dir / "tracked_matanyone_body_refine" / "debug"
                    for slot, (source_path, raw_alpha, frame) in enumerate(
                        zip(output_paths, selected_alpha, output_frames),
                        start=1,
                    ):
                        with Image.open(source_path) as source:
                            instance = tracked_frame_to_instance_result(frame, source_path=source_path, image_size=source.size)
                            body_refine = apply_matanyone_alpha_constraints(
                                source.convert("RGB"),
                                np.asarray(raw_alpha, dtype=np.uint8),
                                instance,
                                body_cfg,
                            )
                        refined = refine_subject_edge(body_refine.image, edge_cfg).image
                        restored = apply_matanyone_alpha_constraints(
                            refined.convert("RGB"),
                            np.asarray(refined.getchannel("A"), dtype=np.uint8),
                            instance,
                            body_cfg,
                        )
                        body_images.append(restored.image)
                        body_metrics.append(_body_refine_metrics(restored))
                        _save_body_refine_debug(restored, output_dir=debug_dir, stem=f"{slot:02d}")
                    body_summary = _save_branch_outputs(
                        "tracked_matanyone_body_refine",
                        body_images,
                        output_frames,
                        output_dir,
                        resolved_compose_final,
                        extra_metrics=body_metrics,
                    )
                    cutout_sheet_rows.append(body_summary.pop("_cutout_sheet_items"))
                    final_sheet_rows.append(body_summary.pop("_final_sheet_items"))
                    body_summary["edge_temporal_jitter"] = compute_edge_temporal_jitter(
                        body_images, output_frames
                    )
                    body_summary["elapsed_seconds"] = time.perf_counter() - matanyone_started
                    branches["tracked_matanyone_body_refine"] = body_summary
                except Exception as exc:
                    branches["tracked_matanyone_body_refine"] = {
                        "status": "failed",
                        "error": str(exc),
                        "elapsed_seconds": time.perf_counter() - matanyone_started,
                    }
            except Exception as exc:
                branches["tracked_matanyone_current"] = {
                    "status": "failed",
                    "error": str(exc),
                    "elapsed_seconds": time.perf_counter() - matanyone_started,
                }
                branches["tracked_matanyone_subject_priority"] = {
                    "status": "failed",
                    "error": str(exc),
                    "elapsed_seconds": time.perf_counter() - matanyone_started,
                }
                branches["tracked_matanyone_body_refine"] = {
                    "status": "failed",
                    "error": str(exc),
                    "elapsed_seconds": time.perf_counter() - matanyone_started,
                }

    _save_sheet_grid(cutout_sheet_rows, output_dir / "sheets" / "cutout_sheet.jpg")
    _save_sheet_grid(final_sheet_rows, output_dir / "sheets" / "final_sheet.jpg")
    summary = {
        "status": "ok",
        "manifest_path": str(manifest_path),
        "elapsed_seconds": time.perf_counter() - started,
        "branches": branches,
    }
    (output_dir / "summary_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def run_session(
    input_root: Path,
    output_root: Path,
    *,
    tracker: Any,
    matanyone: Any | None,
    modnet: Any | None,
    compose_final: Any | None = None,
    edge_config: SubjectEdgeRefineConfig | None = None,
) -> dict[str, Any]:
    manifests = sorted(Path(input_root).rglob("metadata.json"))
    session: dict[str, Any] = {"input_root": str(input_root), "takes": {}, "take_count": len(manifests)}
    for manifest_path in manifests:
        reset_tracker = getattr(tracker, "reset", None)
        if callable(reset_tracker):
            reset_tracker()
        relative = manifest_path.parent.relative_to(input_root)
        take_output = output_root / relative
        session["takes"][str(relative).replace("\\", "/")] = run_take(
            manifest_path,
            take_output,
            tracker=tracker,
            matanyone=matanyone,
            modnet=modnet,
            compose_final=compose_final,
            edge_config=edge_config,
        )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "session_summary.json").write_text(
        json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return session


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate tracked burst video matting offline.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--manifest-path", type=Path)
    input_group.add_argument("--input-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--yolo-seg-model-path", default="models/yolo11x-seg.pt")
    parser.add_argument("--include-modnet", action="store_true")
    parser.add_argument("--include-matanyone", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_root or Path("generated") / "tracked_video_matting_eval" / datetime.now().strftime("%Y%m%d_%H%M%S")
    instance_config = InstanceSegmentationConfig(model_path=args.yolo_seg_model_path)
    tracker = SubjectInstanceTracker(
        instance_config=instance_config,
        tracking_config=TrackingConfig(
            model_path=args.yolo_seg_model_path,
            mask_threshold=instance_config.mask_threshold,
        ),
    )
    from config_manager import load_config

    edge_config = SubjectEdgeRefineConfig.from_mapping(load_config().get("subject_edge_refine", {}))
    modnet = None
    matanyone = None
    if args.include_modnet:
        from modnet_matting_service import ModnetMattingService

        modnet = ModnetMattingService()
    if args.include_matanyone:
        from matanyone_service import MatAnyoneService

        matanyone = MatAnyoneService()
    if args.input_root is not None:
        summary = run_session(
            args.input_root,
            output_dir,
            tracker=tracker,
            matanyone=matanyone,
            modnet=modnet,
            edge_config=edge_config,
        )
    else:
        summary = run_take(
            args.manifest_path,
            output_dir,
            tracker=tracker,
            matanyone=matanyone,
            modnet=modnet,
            edge_config=edge_config,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
