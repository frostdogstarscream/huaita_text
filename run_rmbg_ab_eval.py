from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw

from ali_segment_service import AliSegmentService
from app_state import APP_STATE, OUTPUT_DIR, BASE_DIR
from background_manager import get_background_items
from config_manager import load_config
from image_composer import _place_subject_on_background
from rmbg_segment_service import RmbgSegmentService
from slogan_manager import get_rotation_snapshot
from subject_alpha_filter import SubjectAlphaFilterConfig, filter_subject_alpha, save_alpha_filter_debug
from subject_edge_refine import SubjectEdgeRefineConfig, effective_alpha_bbox, refine_subject_edge, save_edge_refine_debug
from subject_locator import SubjectLocator, SubjectLocatorConfig
from subject_temporal_fusion import TemporalSubjectFusionConfig, fuse_subjects_temporally
from subject_visitor_suppression import (
    SubjectVisitorSuppressionConfig,
    apply_post_alpha_hard_clear,
    build_visitor_suppression_mask,
    suppress_visitors_in_roi,
)
from text_renderer import draw_slogan


@dataclass
class FrameEvalResult:
    cutout: Image.Image
    location: Any | None
    metrics: dict[str, Any]
    elapsed_seconds: float
    error: str | None = None


def _compose_final(subject: Image.Image, background_item: dict[str, Any], slogan: str, slogan_row: int | None) -> Image.Image:
    output_cfg = APP_STATE["config"]["output"]
    target_size = (int(output_cfg["width"]), int(output_cfg["height"]))
    background = _place_subject_on_background(subject, background_item, target_size)
    return draw_slogan(background, slogan, background_item, slogan_row)


def _count_fragments(alpha: np.ndarray, threshold: int = 16) -> int:
    mask = (alpha > threshold).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return 0
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    fragments = 0
    for label in range(1, count):
        if label != largest:
            fragments += 1
    return fragments


def _compute_metrics(cutout: Image.Image, location: Any | None, visitor_cfg: SubjectVisitorSuppressionConfig) -> dict[str, Any]:
    rgba = cutout.convert("RGBA")
    alpha = np.array(rgba.getchannel("A"), dtype=np.uint8)
    h, w = alpha.shape
    fg_mask = alpha > 16
    fg_pixels = int(np.count_nonzero(fg_mask))
    bbox = effective_alpha_bbox(rgba, alpha_threshold=16)
    bbox_area_ratio = 0.0
    if bbox:
        bw = max(0, bbox[2] - bbox[0])
        bh = max(0, bbox[3] - bbox[1])
        bbox_area_ratio = float(bw * bh) / float(w * h)

    visitor_ratio = 0.0
    if location is not None and location.other_person_bboxes:
        visitor_mask, _boxes, _protect = build_visitor_suppression_mask(location, rgba.size, visitor_cfg)
        visitor_pixels = int(np.count_nonzero(visitor_mask))
        if visitor_pixels > 0:
            visitor_fg = int(np.count_nonzero((alpha > 16) & visitor_mask))
            visitor_ratio = float(visitor_fg) / float(visitor_pixels)

    right_top_mask = np.zeros_like(fg_mask)
    right_top_mask[: int(h * 0.45), int(w * 0.65) :] = True
    right_top_total = int(np.count_nonzero(right_top_mask))
    right_top_fg = int(np.count_nonzero(fg_mask & right_top_mask))
    right_top_ratio = float(right_top_fg) / float(max(right_top_total, 1))

    return {
        "foreground_px": fg_pixels,
        "foreground_ratio": float(fg_pixels) / float(w * h),
        "effective_bbox_area_ratio": bbox_area_ratio,
        "visitor_residual_ratio": visitor_ratio,
        "right_top_risk_ratio": right_top_ratio,
        "fragment_count": _count_fragments(alpha, threshold=16),
    }


def _run_rmbg_frame(
    service: RmbgSegmentService,
    capture_path: Path,
    shot_id: str,
    cutout_path: Path,
    locator: SubjectLocator,
    alpha_cfg: SubjectAlphaFilterConfig,
    visitor_cfg: SubjectVisitorSuppressionConfig,
    edge_cfg: SubjectEdgeRefineConfig,
) -> FrameEvalResult:
    t0 = time.perf_counter()
    location = locator.locate(capture_path, shot_id)
    segment_source = capture_path
    if location is not None:
        segment_source = location.roi_path
        if visitor_cfg.enabled and visitor_cfg.pre_aliyun_enabled:
            suppression = suppress_visitors_in_roi(
                location.roi_path,
                location,
                output_dir=location.roi_path.parent,
                stem=shot_id,
                config=visitor_cfg,
            )
            segment_source = suppression.cleaned_roi_path

    image = service.segment_image_file(segment_source, cutout_path)
    if location is not None and alpha_cfg.enabled:
        image = filter_subject_alpha(image, location, alpha_cfg)
        save_alpha_filter_debug(image, location, OUTPUT_DIR / "subject_debug" / f"{shot_id}_rmbg_alpha_filter.png", alpha_cfg)
    if location is not None and visitor_cfg.enabled:
        image = apply_post_alpha_hard_clear(image, location, visitor_cfg)
    if edge_cfg.enabled:
        refine_result = refine_subject_edge(image, edge_cfg)
        save_edge_refine_debug(
            image,
            refine_result,
            output_dir=OUTPUT_DIR / "subject_debug",
            stem=f"{shot_id}_rmbg",
            config=edge_cfg,
        )
        image = refine_result.image

    image.save(cutout_path, format="PNG")
    metrics = _compute_metrics(image, location, visitor_cfg)
    return FrameEvalResult(
        cutout=image,
        location=location,
        metrics=metrics,
        elapsed_seconds=time.perf_counter() - t0,
    )


def _run_aliyun_frame(
    service: AliSegmentService,
    capture_path: Path,
    shot_id: str,
    cutout_path: Path,
    visitor_cfg: SubjectVisitorSuppressionConfig,
) -> FrameEvalResult:
    t0 = time.perf_counter()
    image = service.segment_image_file(capture_path, cutout_path)
    location = None
    locator = getattr(service, "subject_locator", None)
    if locator is not None:
        location = locator.locate(capture_path, f"{shot_id}_metrics")
    metrics = _compute_metrics(image, location, visitor_cfg)
    return FrameEvalResult(
        cutout=image,
        location=location,
        metrics=metrics,
        elapsed_seconds=time.perf_counter() - t0,
    )


def _save_compare_sheet(images: list[Image.Image], output_path: Path, labels: list[str]) -> None:
    thumb_w = 240
    thumb_h = 330
    gap = 14
    cols = len(images)
    canvas = Image.new("RGB", (cols * thumb_w + (cols + 1) * gap, thumb_h + 48), (245, 239, 223))
    draw = ImageDraw.Draw(canvas)
    for i, (image, label) in enumerate(zip(images, labels)):
        x = gap + i * (thumb_w + gap)
        y = 36
        fit = ImageOps.contain(image.convert("RGBA"), (thumb_w, thumb_h))
        tile = Image.new("RGBA", (thumb_w, thumb_h), (255, 255, 255, 255))
        tile.paste(fit, ((thumb_w - fit.width) // 2, (thumb_h - fit.height) // 2), fit)
        canvas.paste(tile.convert("RGB"), (x, y))
        draw.text((x + 6, 8), label, fill=(30, 30, 30))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="JPEG", quality=92)


def run_ab_eval(group_ids: list[str], captures_dir: Path) -> Path:
    APP_STATE["config"] = load_config()
    config = APP_STATE["config"]
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = OUTPUT_DIR / "ab_eval_rmbg" / now
    out_root.mkdir(parents=True, exist_ok=True)

    locator = SubjectLocator(
        SubjectLocatorConfig.from_mapping(config.get("subject_locator", {})),
        output_dir=OUTPUT_DIR / "subject_rois",
    )
    alpha_cfg = SubjectAlphaFilterConfig.from_mapping(config.get("subject_alpha_filter", {}))
    visitor_cfg = SubjectVisitorSuppressionConfig.from_mapping(config.get("subject_visitor_suppression", {}))
    edge_cfg = SubjectEdgeRefineConfig.from_mapping(config.get("subject_edge_refine", {}))
    temporal_cfg = TemporalSubjectFusionConfig.from_mapping(config.get("temporal_subject_fusion", {}))

    aliyun_service = AliSegmentService(config)
    rmbg_service = RmbgSegmentService(prefer_cuda=True)

    snapshot = get_rotation_snapshot()
    draw_slogan_text = snapshot.get("slogan_content") or snapshot["slogan"]
    slogan_row = int(snapshot.get("slogan_row", 1))
    background_item = get_background_items()[0]

    summary: dict[str, Any] = {"groups": {}, "generated_at": now, "rmbg_device": rmbg_service.runtime.device}
    for group_id in group_ids:
        captures = [captures_dir / f"{group_id}_{i}.jpg" for i in range(1, 5)]
        if not all(path.exists() for path in captures):
            raise FileNotFoundError(f"Missing captures for group {group_id}: {captures}")

        group_result: dict[str, Any] = {}
        for engine in ("aliyun", "rmbg"):
            frame_results: list[FrameEvalResult] = []
            engine_cutout_dir = out_root / engine / "cutouts" / group_id
            engine_final_dir = out_root / engine / "final" / group_id
            engine_cutout_dir.mkdir(parents=True, exist_ok=True)
            engine_final_dir.mkdir(parents=True, exist_ok=True)

            start_engine = time.perf_counter()
            for index, capture_path in enumerate(captures, start=1):
                shot_id = f"{group_id}_{index}"
                cutout_path = engine_cutout_dir / f"{shot_id}.png"
                try:
                    if engine == "aliyun":
                        result = _run_aliyun_frame(
                            aliyun_service,
                            capture_path,
                            shot_id,
                            cutout_path,
                            visitor_cfg,
                        )
                    else:
                        result = _run_rmbg_frame(
                            rmbg_service,
                            capture_path,
                            shot_id,
                            cutout_path,
                            locator,
                            alpha_cfg,
                            visitor_cfg,
                            edge_cfg,
                        )
                except Exception as exc:
                    fallback = Image.open(capture_path).convert("RGBA")
                    fallback.save(cutout_path, format="PNG")
                    result = FrameEvalResult(
                        cutout=fallback,
                        location=None,
                        metrics=_compute_metrics(fallback, None, visitor_cfg),
                        elapsed_seconds=0.0,
                        error=str(exc),
                    )
                frame_results.append(result)

            cutouts = [item.cutout for item in frame_results]
            fused_cutouts, fusion_report = fuse_subjects_temporally(
                cutouts,
                temporal_cfg,
                debug_dir=out_root / engine / "debug",
                debug_stem=group_id,
            )
            for idx, fused in enumerate(fused_cutouts, start=1):
                if fused is None:
                    continue
                fused.save(engine_cutout_dir / f"{group_id}_{idx}_fused.png", format="PNG")
                final = _compose_final(fused, background_item, draw_slogan_text, slogan_row)
                final.save(engine_final_dir / f"{group_id}_{idx}.jpg", format="JPEG", quality=92)

            single_sheet = out_root / f"{group_id}_{engine}_single_sheet.jpg"
            final_sheet = out_root / f"{group_id}_{engine}_final_sheet.jpg"
            _save_compare_sheet(cutouts, single_sheet, [f"{engine}-S{i}" for i in range(1, 5)])
            final_images = [Image.open(engine_final_dir / f"{group_id}_{i}.jpg").convert("RGB") for i in range(1, 5)]
            _save_compare_sheet(final_images, final_sheet, [f"{engine}-F{i}" for i in range(1, 5)])

            elapsed = time.perf_counter() - start_engine
            frame_metrics = [item.metrics for item in frame_results]
            group_result[engine] = {
                "elapsed_seconds": elapsed,
                "fusion": {
                    "alignment_success_count": fusion_report.alignment_success_count,
                    "alpha_stable_ratio": fusion_report.alpha_stable_ratio,
                    "removed_temporal_noise_px": fusion_report.removed_temporal_noise_px,
                    "fallback_reason": fusion_report.fallback_reason,
                },
                "frame_metrics": frame_metrics,
                "errors": [item.error for item in frame_results if item.error],
                "avg_visitor_residual_ratio": float(np.mean([m["visitor_residual_ratio"] for m in frame_metrics])),
                "avg_right_top_risk_ratio": float(np.mean([m["right_top_risk_ratio"] for m in frame_metrics])),
                "avg_effective_bbox_area_ratio": float(np.mean([m["effective_bbox_area_ratio"] for m in frame_metrics])),
                "avg_fragment_count": float(np.mean([m["fragment_count"] for m in frame_metrics])),
            }

        summary["groups"][group_id] = group_result

        cmp_images: list[Image.Image] = []
        labels: list[str] = []
        for engine in ("aliyun", "rmbg"):
            for i in range(1, 5):
                cmp_images.append(Image.open(out_root / engine / "final" / group_id / f"{group_id}_{i}.jpg").convert("RGB"))
                labels.append(f"{engine}-{i}")
        _save_compare_sheet(cmp_images, out_root / f"{group_id}_final_compare_sheet.jpg", labels)

    summary_path = out_root / "summary_metrics.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[AB-EVAL] done: {summary_path}")
    return out_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline A/B evaluation: YOLO+Aliyun vs YOLO+RMBG")
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["8032532334c940d28cf78782fc2d43b3", "9595dd5a6d504901a8f6911a9a951353"],
        help="Capture group ids",
    )
    parser.add_argument(
        "--captures-dir",
        default=str(BASE_DIR / "generated" / "captures"),
        help="Directory containing capture jpg files",
    )
    args = parser.parse_args()
    run_ab_eval(args.groups, Path(args.captures_dir))


if __name__ == "__main__":
    main()
