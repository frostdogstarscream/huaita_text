"""Image composition: subject placement, background compositing, and final output."""

import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote

from PIL import Image

from ali_segment_service import AliSegmentError, AliSegmentService
from app_state import APP_STATE, FINAL_DIR, CUTOUT_DIR, RESOURCE_DIR
from background_manager import get_background_items, resolve_output_size, resolve_person_layout, select_rotating_background
from text_renderer import draw_slogan
from subject_edge_refine import effective_alpha_bbox


def resize_cover(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    target_w, target_h = target_size
    src_w, src_h = image.size
    scale = max(target_w / src_w, target_h / src_h)
    new_w = max(int(src_w * scale), 1)
    new_h = max(int(src_h * scale), 1)
    resized = image.resize((new_w, new_h), Image.LANCZOS)
    left = max((new_w - target_w) // 2, 0)
    top = max((new_h - target_h) // 2, 0)
    return resized.crop((left, top, left + target_w, top + target_h))


def effective_subject_bbox(image: Image.Image, alpha_threshold: int = 16) -> tuple[int, int, int, int] | None:
    return effective_alpha_bbox(image, alpha_threshold=alpha_threshold)


def build_subject_cutout(capture_path: Path, task_id: str) -> Image.Image:
    matting_service = APP_STATE["matting_service"]
    if matting_service is None:
        provider = str(APP_STATE.get("config", {}).get("matting_api", {}).get("provider", "unknown"))
        raise AliSegmentError(f"matting_service_not_initialized: provider={provider}")
    cutout_path = CUTOUT_DIR / f"{task_id}.png"
    image = matting_service.segment_image_file(capture_path, cutout_path)
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    if not image.getbbox():
        raise AliSegmentError("segment result is empty.")
    return image


def _place_subject_on_background(
    subject: Image.Image,
    background_item: dict[str, Any],
    target_size: tuple[int, int],
    subject_bbox: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    if subject_bbox is not None:
        left, top, right, bottom = subject_bbox
        left = max(int(left), 0)
        top = max(int(top), 0)
        right = min(int(right), subject.width)
        bottom = min(int(bottom), subject.height)
        if right <= left or bottom <= top:
            raise ValueError("subject_bbox is empty or out of bounds.")
        bbox = (left, top, right, bottom)
    else:
        edge_cfg = APP_STATE.get("config", {}).get("subject_edge_refine", {})
        alpha_threshold = int(edge_cfg.get("effective_bbox_alpha_threshold", 16))
        bbox = effective_subject_bbox(subject, alpha_threshold=alpha_threshold)
    if not bbox:
        raise ValueError("No foreground detected in segmented image.")
    subject = subject.crop(bbox)

    background_path = RESOURCE_DIR / background_item["path"]
    if not background_path.exists():
        raise FileNotFoundError(f"Background not found: {background_path}")

    person_layout = resolve_person_layout(background_item)
    target_height = max(int(target_size[1] * float(person_layout["target_height_ratio"])), 1)
    center_x_ratio = float(person_layout["center_x_ratio"])
    center_y_offset = int(person_layout["center_y_offset"])
    bottom_margin = int(person_layout["bottom_margin"])
    max_width_ratio = float(person_layout.get("max_width_ratio", 1.0))
    max_width = max(int(target_size[0] * max_width_ratio), 1)
    height_scale = target_height / max(subject.height, 1)
    width_scale = max_width / max(subject.width, 1)
    scale = min(height_scale, width_scale)
    resized_subject = subject.resize(
        (max(int(subject.width * scale), 1), max(int(subject.height * scale), 1)),
        Image.LANCZOS,
    )

    background = resize_cover(Image.open(background_path).convert("RGBA"), target_size)
    center_x = int(background.width * center_x_ratio)
    x = center_x - (resized_subject.width // 2)
    y = background.height - resized_subject.height - bottom_margin + center_y_offset
    max_x = max(background.width - resized_subject.width, 0)
    max_y = max(background.height - resized_subject.height, 0)
    x = max(0, min(x, max_x))
    y = max(0, min(y, max_y))
    background.paste(resized_subject, (x, y), resized_subject)
    return background


def compose_single_variant(
    subject: Image.Image,
    slogan: str,
    task_id: str,
    background_item: dict[str, Any],
    order: int,
    slogan_row: int | None = None,
) -> dict[str, Any]:
    cfg = APP_STATE["config"]
    output_cfg = cfg["output"]
    target_size = resolve_output_size(background_item)

    background = _place_subject_on_background(subject, background_item, target_size)
    final_image = draw_slogan(background, slogan, background_item, slogan_row)

    image_id = uuid.uuid4().hex[:12]
    filename = f"{task_id}_{order}_{image_id}.jpg"
    output_path = FINAL_DIR / filename
    final_image.convert("RGB").save(output_path, quality=int(output_cfg["jpeg_quality"]))
    return {
        "image_id": image_id,
        "image_url": f"/generated/final/{quote(filename)}",
        "background_id": background_item["id"],
        "background_name": background_item["name"],
        "orientation": background_item.get("orientation", "portrait"),
        "order": order,
    }


def compose_variants(
    subject: Image.Image,
    slogan: str,
    task_id: str,
    repeats_per_background: int = 1,
    slogan_row: int | None = None,
) -> list[dict[str, Any]]:
    cfg = APP_STATE["config"]
    output_cfg = cfg["output"]

    edge_cfg = APP_STATE.get("config", {}).get("subject_edge_refine", {})
    alpha_threshold = int(edge_cfg.get("effective_bbox_alpha_threshold", 16))
    bbox = effective_subject_bbox(subject, alpha_threshold=alpha_threshold)
    if not bbox:
        raise ValueError("No foreground detected in segmented image.")
    subject = subject.crop(bbox)

    results: list[dict[str, Any]] = []
    order = 1
    repeats = max(int(repeats_per_background), 1)
    for item in get_background_items():
        target_size = resolve_output_size(item)
        background = _place_subject_on_background(subject, item, target_size)
        final_image = draw_slogan(background, slogan, item, slogan_row)

        for _ in range(repeats):
            image_id = uuid.uuid4().hex[:12]
            filename = f"{task_id}_{order}_{image_id}.jpg"
            output_path = FINAL_DIR / filename
            final_image.convert("RGB").save(output_path, quality=int(output_cfg["jpeg_quality"]))
            results.append(
                {
                    "image_id": image_id,
                    "image_url": f"/generated/final/{quote(filename)}",
                    "background_id": item["id"],
                    "background_name": item["name"],
                    "orientation": item.get("orientation", "portrait"),
                    "order": order,
                }
            )
            order += 1
    return results
