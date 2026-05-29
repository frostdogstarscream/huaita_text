from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from app_state import BASE_DIR, OUTPUT_DIR
from subject_instance_segmentation import (
    InstanceSegmentationConfig,
    InstanceSegmentationResult,
    SubjectInstanceSegmenter,
    cutout_from_instance_mask,
)


DEFAULT_GROUPS = [
    "8032532334c940d28cf78782fc2d43b3",
    "9595dd5a6d504901a8f6911a9a951353",
]


def _component_count(mask: np.ndarray) -> int:
    count, _labels, _stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return max(count - 1, 0)


def _metrics(result: InstanceSegmentationResult | None, elapsed_seconds: float, error: str | None) -> dict[str, Any]:
    if result is None:
        return {
            "ok": False,
            "error": error,
            "elapsed_seconds": elapsed_seconds,
        }
    visitor_mask = np.zeros((result.image_size[1], result.image_size[0]), dtype=bool)
    for visitor in result.visitors:
        visitor_mask |= visitor.mask.astype(bool)

    selected_mask = result.selected.mask.astype(bool)
    overlap_px = int(np.count_nonzero(selected_mask & visitor_mask))
    visitor_px = int(np.count_nonzero(visitor_mask))
    selected_px = int(np.count_nonzero(selected_mask))
    unknown_px = int(np.count_nonzero(result.unknown))
    sure_fg_px = int(np.count_nonzero(result.sure_foreground))
    sure_bg_px = int(np.count_nonzero(result.sure_background))

    return {
        "ok": True,
        "elapsed_seconds": elapsed_seconds,
        "candidate_count": len(result.candidates),
        "visitor_count": len(result.visitors),
        "selected_score": result.selected.score,
        "selected_confidence": result.selected.confidence,
        "selected_mask_px": selected_px,
        "visitor_mask_px": visitor_px,
        "selected_visitor_overlap_ratio": float(overlap_px) / float(max(visitor_px, 1)),
        "trimap_unknown_px": unknown_px,
        "trimap_sure_fg_px": sure_fg_px,
        "trimap_sure_bg_px": sure_bg_px,
        "subject_component_count": _component_count(selected_mask),
        "visitor_component_count": _component_count(visitor_mask),
    }


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


def run_eval(
    *,
    groups: list[str],
    captures_dir: Path,
    model_path: str,
    output_root: Path | None = None,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = output_root or (OUTPUT_DIR / "yolo_seg_instance_eval" / timestamp)
    out_root.mkdir(parents=True, exist_ok=True)

    config = InstanceSegmentationConfig(model_path=model_path)
    segmenter = SubjectInstanceSegmenter(config, output_dir=out_root / "debug")
    summary: dict[str, Any] = {
        "model_path": model_path,
        "generated_at": timestamp,
        "groups": {},
        "matting_status": "not_run: this runner validates instance ownership and trimap constraints first",
    }

    for group in groups:
        group_metrics: list[dict[str, Any]] = []
        debug_items: list[tuple[Image.Image, str]] = []
        cutout_items: list[tuple[Image.Image, str]] = []
        for index in range(1, 5):
            capture_path = captures_dir / f"{group}_{index}.jpg"
            stem = f"{group}_{index}"
            if not capture_path.exists():
                group_metrics.append(_metrics(None, 0.0, f"missing capture: {capture_path}"))
                continue

            t0 = time.perf_counter()
            result = segmenter.segment(capture_path, stem)
            elapsed = time.perf_counter() - t0
            group_metrics.append(_metrics(result, elapsed, None if result is not None else "segmentation failed"))

            debug_path = out_root / "debug" / f"{stem}_seg_instance_debug.jpg"
            if debug_path.exists():
                debug_items.append((Image.open(debug_path).convert("RGB"), f"debug-{index}"))
            if result is not None:
                cutout = cutout_from_instance_mask(capture_path, result.selected)
                cutout_path = out_root / "cutouts" / f"{stem}_yolo_seg_cutout.png"
                cutout_path.parent.mkdir(parents=True, exist_ok=True)
                cutout.save(cutout_path, format="PNG")
                cutout_items.append((cutout, f"mask-{index}"))

        if debug_items:
            _save_sheet(debug_items, out_root / f"{group}_seg_instance_debug_sheet.jpg")
        if cutout_items:
            _save_sheet(cutout_items, out_root / f"{group}_yolo_seg_cutout_sheet.jpg")

        ok_metrics = [item for item in group_metrics if item.get("ok")]
        summary["groups"][group] = {
            "frames": group_metrics,
            "ok_count": len(ok_metrics),
            "avg_elapsed_seconds": float(np.mean([item["elapsed_seconds"] for item in ok_metrics])) if ok_metrics else 0.0,
            "avg_visitor_count": float(np.mean([item["visitor_count"] for item in ok_metrics])) if ok_metrics else 0.0,
            "avg_selected_visitor_overlap_ratio": float(np.mean([item["selected_visitor_overlap_ratio"] for item in ok_metrics])) if ok_metrics else 0.0,
        }

    summary_path = out_root / "summary_metrics.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[YOLO-SEG-EVAL] done: {summary_path}")
    return out_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline YOLO-seg subject/visitor instance ownership evaluation.")
    parser.add_argument("--groups", nargs="+", default=DEFAULT_GROUPS)
    parser.add_argument("--captures-dir", default=str(BASE_DIR / "generated" / "captures"))
    parser.add_argument("--model-path", default=str(BASE_DIR / "models" / "yolo11x-seg.pt"))
    args = parser.parse_args()
    run_eval(groups=args.groups, captures_dir=Path(args.captures_dir), model_path=args.model_path)


if __name__ == "__main__":
    main()
