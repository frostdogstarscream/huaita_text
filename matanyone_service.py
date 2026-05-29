from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from subject_instance_segmentation import InstanceSegmentationResult


class MatAnyoneError(RuntimeError):
    def __init__(self, message: str, *, stage: str = "matting") -> None:
        super().__init__(message)
        self.stage = stage


@dataclass(frozen=True)
class MatAnyoneConstraintConfig:
    enabled: bool = True
    initial_mask_mode: str = "subject_instance_mask"
    core_erode_px: int = 6
    body_soft_band_px: int = 5
    head_soft_band_px: int = 12
    head_height_ratio: float = 0.34
    visitor_clear_dilate_px: int = 8
    hair_side_refine_enabled: bool = True
    hair_refine_side: str = "right"
    hair_refine_height_ratio: float = 0.28
    hair_refine_inner_rejudge_px: int = 5
    hair_refine_outer_soft_band_px: int = 4
    hair_refine_min_alpha: int = 16
    body_refine_policy: str = "aggressive"
    occlusion_conflict_policy: str = "visitor_priority"
    contact_subject_priority_enabled: bool = False
    contact_visitor_dilate_px: int = 8
    contact_core_erode_px: int = 6
    contact_edge_alpha_floor: int = 160
    torso_refine_enabled: bool = True
    torso_inner_rejudge_px: int = 4
    torso_outer_soft_band_px: int = 1
    arm_refine_enabled: bool = True
    arm_inner_rejudge_px: int = 3
    arm_outer_soft_band_px: int = 1
    body_contact_edge_alpha_floor: int = 190
    debug_enabled: bool = True

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "MatAnyoneConstraintConfig":
        if not isinstance(raw, dict):
            return cls()
        values = {}
        for field_name in cls.__dataclass_fields__:
            if field_name in raw:
                values[field_name] = raw[field_name]
        return cls(**values)

    def with_body_match_head(self) -> "MatAnyoneConstraintConfig":
        body_outer = max(int(self.hair_refine_outer_soft_band_px), 0)
        return replace(
            self,
            body_refine_policy="match_head",
            body_soft_band_px=max(int(self.head_soft_band_px), 0),
            torso_inner_rejudge_px=0,
            arm_inner_rejudge_px=0,
            torso_outer_soft_band_px=body_outer,
            arm_outer_soft_band_px=body_outer,
            body_contact_edge_alpha_floor=int(self.contact_edge_alpha_floor),
        )


@dataclass(frozen=True)
class MatAnyoneConstraintMasks:
    allowed_support: np.ndarray
    forced_foreground: np.ndarray
    forced_background: np.ndarray
    soft_unknown: np.ndarray
    visitor_clear: np.ndarray
    hair_region: np.ndarray
    hair_inner_rejudge: np.ndarray
    hair_outer_support: np.ndarray
    contact_conflict: np.ndarray
    contact_core: np.ndarray
    contact_edge: np.ndarray
    visitor_visible_clear: np.ndarray
    torso_region: np.ndarray
    arm_region: np.ndarray
    body_inner_rejudge: np.ndarray
    body_outer_support: np.ndarray


@dataclass(frozen=True)
class MatAnyoneConstraintResult:
    image: Image.Image
    raw_alpha: np.ndarray
    constrained_alpha: np.ndarray
    masks: MatAnyoneConstraintMasks
    raw_foreground_px: int
    constrained_foreground_px: int
    outside_subject_alpha_px: int
    outside_subject_alpha_ratio: float
    soft_band_alpha_px: int
    right_hair_outside_alpha_px: int
    right_hair_removed_alpha_px: int
    right_hair_retained_alpha_px: int
    contact_conflict_px: int
    contact_core_restored_px: int
    contact_edge_floor_applied_px: int
    visitor_visible_residual_ratio: float
    subject_contact_missing_ratio: float
    body_outside_soft_alpha_ratio: float
    body_edge_removed_px: int
    body_core_missing_ratio: float


def _morph(mask: np.ndarray, operation: int, radius: int) -> np.ndarray:
    if int(radius) <= 0:
        return mask.astype(bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(radius) * 2 + 1, int(radius) * 2 + 1))
    return cv2.morphologyEx(mask.astype(np.uint8), operation, kernel).astype(bool)


def build_matanyone_initial_mask(
    result: InstanceSegmentationResult,
    config: MatAnyoneConstraintConfig | None = None,
) -> np.ndarray:
    cfg = config or MatAnyoneConstraintConfig()
    if str(cfg.initial_mask_mode).lower() == "bbox":
        mask = np.zeros_like(result.selected.mask, dtype=np.uint8)
        height, width = mask.shape
        left, top, right, bottom = (int(round(value)) for value in result.selected.bbox)
        mask[max(0, top):min(height, bottom), max(0, left):min(width, right)] = 255
        return mask
    return result.selected.mask.astype(np.uint8) * 255


def build_matanyone_constraint_masks(
    result: InstanceSegmentationResult,
    config: MatAnyoneConstraintConfig | None = None,
) -> MatAnyoneConstraintMasks:
    cfg = config or MatAnyoneConstraintConfig()
    if str(cfg.body_refine_policy).lower() == "match_head":
        cfg = cfg.with_body_match_head()
    subject = result.selected.mask.astype(bool)
    empty = np.zeros_like(subject, dtype=bool)
    if not cfg.enabled:
        return MatAnyoneConstraintMasks(
            np.ones_like(subject, dtype=bool),
            empty,
            empty,
            np.ones_like(subject, dtype=bool),
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
        )

    core = _morph(subject, cv2.MORPH_ERODE, int(cfg.core_erode_px))
    body_support = _morph(subject, cv2.MORPH_DILATE, int(cfg.body_soft_band_px))
    head_subject = np.zeros_like(subject, dtype=bool)
    _left, top, _right, bottom = result.selected.bbox
    head_bottom = int(round(float(top) + max(float(bottom) - float(top), 1.0) * float(cfg.head_height_ratio)))
    head_subject[max(0, int(np.floor(top))):min(subject.shape[0], head_bottom), :] = subject[
        max(0, int(np.floor(top))):min(subject.shape[0], head_bottom), :
    ]
    head_support = _morph(head_subject, cv2.MORPH_DILATE, int(cfg.head_soft_band_px))
    allowed_support = body_support | head_support

    hair_region = empty.copy()
    hair_inner_rejudge = empty.copy()
    hair_outer_support = empty.copy()
    torso_region = empty.copy()
    arm_region = empty.copy()
    body_inner_rejudge = empty.copy()
    body_outer_support = empty.copy()
    if cfg.hair_side_refine_enabled:
        left, top, right, bottom = (float(value) for value in result.selected.bbox)
        hair_bottom = int(round(top + max(bottom - top, 1.0) * float(cfg.hair_refine_height_ratio)))
        side = str(cfg.hair_refine_side).lower()
        side_mask = np.zeros_like(subject, dtype=bool)
        top_px = max(0, int(np.floor(top)))
        hair_bottom = min(subject.shape[0], hair_bottom)
        hair_slice = subject[top_px:hair_bottom, :]
        _hair_y, hair_x = np.where(hair_slice)
        if hair_x.size:
            head_left = int(hair_x.min())
            head_right = int(hair_x.max()) + 1
            center_x = int(np.ceil((head_left + head_right) / 2.0))
            if side == "left":
                side_mask[top_px:hair_bottom, head_left:max(head_left, center_x)] = True
            else:
                side_mask[top_px:hair_bottom, min(subject.shape[1], center_x):head_right] = True
            hair_region = subject & side_mask
            rejudge_depth = int(cfg.core_erode_px) + int(cfg.hair_refine_inner_rejudge_px)
            subject_boundary = subject & ~_morph(subject, cv2.MORPH_ERODE, rejudge_depth)
            hair_inner_rejudge = subject_boundary & hair_region
            outer_scope_radius = max(int(cfg.head_soft_band_px), int(cfg.body_soft_band_px))
            wide_hair_outer = _morph(hair_region, cv2.MORPH_DILATE, outer_scope_radius) & ~subject
            narrow_hair_outer = _morph(hair_region, cv2.MORPH_DILATE, int(cfg.hair_refine_outer_soft_band_px)) & ~subject
            side_outer = np.zeros_like(subject, dtype=bool)
            if side == "left":
                side_outer[:, :max(0, center_x)] = True
            else:
                side_outer[:, min(subject.shape[1], center_x):] = True
            wide_hair_outer &= side_outer
            hair_outer_support = narrow_hair_outer & side_outer
            allowed_support = (allowed_support & ~wide_hair_outer) | hair_outer_support

    # Body regions (torso + arms) for full-body edge refinement policy.
    left, top, right, bottom = (int(round(value)) for value in result.selected.bbox)
    box_h = max(bottom - top, 1)
    shoulder_y = top + int(round(box_h * 0.30))
    hip_y = top + int(round(box_h * 0.78))
    center_x = int(round((left + right) / 2.0))
    torso_slice = np.zeros_like(subject, dtype=bool)
    torso_slice[max(0, shoulder_y):min(subject.shape[0], hip_y), :] = True
    torso_x_band = np.zeros_like(subject, dtype=bool)
    torso_width = max(int(round((right - left) * 0.72)), 1)
    torso_left = max(left, center_x - torso_width // 2)
    torso_right = min(right, torso_left + torso_width)
    torso_x_band[:, max(0, torso_left):min(subject.shape[1], torso_right)] = True
    torso_region = subject & torso_slice & torso_x_band

    arm_slice = np.zeros_like(subject, dtype=bool)
    arm_slice[max(0, shoulder_y):min(subject.shape[0], bottom), :] = True
    side_band = np.zeros_like(subject, dtype=bool)
    side_margin = max(int(round((right - left) * 0.30)), 1)
    side_band[:, max(0, left - 1):min(subject.shape[1], left + side_margin)] = True
    side_band[:, max(0, right - side_margin):min(subject.shape[1], right + 1)] = True
    arm_region = subject & arm_slice & side_band

    body_regions = np.zeros_like(subject, dtype=bool)
    if cfg.torso_refine_enabled:
        body_regions |= torso_region
    if cfg.arm_refine_enabled:
        body_regions |= arm_region
    if np.any(body_regions):
        use_body_inner_rejudge = not (
            str(cfg.body_refine_policy).lower() == "match_head"
            and int(cfg.torso_inner_rejudge_px) <= 0
            and int(cfg.arm_inner_rejudge_px) <= 0
        )
        if use_body_inner_rejudge:
            body_boundary = subject & ~_morph(
                subject,
                cv2.MORPH_ERODE,
                int(cfg.core_erode_px) + max(int(cfg.torso_inner_rejudge_px), int(cfg.arm_inner_rejudge_px)),
            )
            body_inner_rejudge = body_boundary & body_regions
        wide_body_outer = _morph(body_regions, cv2.MORPH_DILATE, max(int(cfg.body_soft_band_px), int(cfg.head_soft_band_px))) & ~subject
        torso_outer = _morph(torso_region, cv2.MORPH_DILATE, int(cfg.torso_outer_soft_band_px)) & ~subject
        arm_outer = _morph(arm_region, cv2.MORPH_DILATE, int(cfg.arm_outer_soft_band_px)) & ~subject
        body_outer_support = np.zeros_like(subject, dtype=bool)
        if cfg.torso_refine_enabled:
            body_outer_support |= torso_outer
        if cfg.arm_refine_enabled:
            body_outer_support |= arm_outer
        allowed_support = (allowed_support & ~wide_body_outer) | body_outer_support

    visitor_clear = np.zeros_like(subject, dtype=bool)
    for visitor in result.visitors:
        visitor_clear |= _morph(visitor.mask.astype(bool), cv2.MORPH_DILATE, int(cfg.visitor_clear_dilate_px))
    contact_conflict = empty.copy()
    contact_core = empty.copy()
    contact_edge = empty.copy()
    visitor_visible_clear = visitor_clear
    subject_priority = (
        cfg.contact_subject_priority_enabled
        and str(cfg.occlusion_conflict_policy).lower() == "selected_subject_priority"
    )
    if subject_priority:
        visitor_clear = np.zeros_like(subject, dtype=bool)
        for visitor in result.visitors:
            visitor_clear |= _morph(
                visitor.mask.astype(bool),
                cv2.MORPH_DILATE,
                int(cfg.contact_visitor_dilate_px),
            )
        contact_conflict = subject & visitor_clear
        contact_subject_core = _morph(subject, cv2.MORPH_ERODE, int(cfg.contact_core_erode_px))
        contact_core = contact_conflict & contact_subject_core
        contact_edge = contact_conflict & ~contact_core
        visitor_visible_clear = visitor_clear & ~subject
        forced_background = (~allowed_support) | visitor_visible_clear
        forced_foreground = (core & ~hair_inner_rejudge & ~body_inner_rejudge) | contact_core
    else:
        forced_background = (~allowed_support) | visitor_clear
        forced_foreground = core & ~visitor_clear & ~hair_inner_rejudge & ~body_inner_rejudge
    soft_unknown = allowed_support & ~forced_foreground & ~forced_background
    return MatAnyoneConstraintMasks(
        allowed_support,
        forced_foreground,
        forced_background,
        soft_unknown,
        visitor_clear,
        hair_region,
        hair_inner_rejudge,
        hair_outer_support,
        contact_conflict,
        contact_core,
        contact_edge,
        visitor_visible_clear,
        torso_region,
        arm_region,
        body_inner_rejudge,
        body_outer_support,
    )


def apply_matanyone_alpha_constraints(
    image: Image.Image,
    raw_alpha: np.ndarray | Image.Image,
    result: InstanceSegmentationResult,
    config: MatAnyoneConstraintConfig | None = None,
) -> MatAnyoneConstraintResult:
    cfg = config or MatAnyoneConstraintConfig()
    alpha = np.array(raw_alpha.convert("L") if isinstance(raw_alpha, Image.Image) else raw_alpha, dtype=np.uint8)
    if alpha.shape != result.selected.mask.shape:
        alpha = cv2.resize(alpha, (result.image_size[0], result.image_size[1]), interpolation=cv2.INTER_LINEAR)
    masks = build_matanyone_constraint_masks(result, config)
    constrained = alpha.copy()
    constrained[masks.forced_background] = 0
    constrained[masks.forced_foreground] = 255
    if cfg.contact_subject_priority_enabled and str(cfg.occlusion_conflict_policy).lower() == "selected_subject_priority":
        constrained[masks.contact_edge] = np.maximum(
            constrained[masks.contact_edge],
            np.uint8(cfg.contact_edge_alpha_floor),
        )
        body_contact_edge = masks.contact_edge & (masks.torso_region | masks.arm_region)
        constrained[body_contact_edge] = np.maximum(
            constrained[body_contact_edge],
            np.uint8(cfg.body_contact_edge_alpha_floor),
        )
    rgba = image.convert("RGBA")
    rgba.putalpha(Image.fromarray(constrained, "L"))
    foreground = constrained > 16
    outside = foreground & ~result.selected.mask.astype(bool)
    foreground_px = int(np.count_nonzero(foreground))
    hair_foreground = constrained > int(cfg.hair_refine_min_alpha)
    visible_visitor_px = int(np.count_nonzero(masks.visitor_visible_clear))
    contact_px = int(np.count_nonzero(masks.contact_conflict))
    body_region = masks.torso_region | masks.arm_region
    body_soft = foreground & masks.soft_unknown & body_region
    body_inner_px = int(np.count_nonzero(masks.body_inner_rejudge))
    core_mask = _morph(result.selected.mask.astype(bool), cv2.MORPH_ERODE, int(cfg.core_erode_px))
    return MatAnyoneConstraintResult(
        image=rgba,
        raw_alpha=alpha,
        constrained_alpha=constrained,
        masks=masks,
        raw_foreground_px=int(np.count_nonzero(alpha > 16)),
        constrained_foreground_px=foreground_px,
        outside_subject_alpha_px=int(np.count_nonzero(outside)),
        outside_subject_alpha_ratio=float(np.count_nonzero(outside)) / max(foreground_px, 1),
        soft_band_alpha_px=int(np.count_nonzero(foreground & masks.soft_unknown)),
        right_hair_outside_alpha_px=int(np.count_nonzero(hair_foreground & masks.hair_outer_support)),
        right_hair_removed_alpha_px=int(np.count_nonzero((constrained <= int(cfg.hair_refine_min_alpha)) & masks.hair_inner_rejudge)),
        right_hair_retained_alpha_px=int(np.count_nonzero(hair_foreground & masks.hair_region)),
        contact_conflict_px=contact_px,
        contact_core_restored_px=int(np.count_nonzero(masks.contact_core & (alpha < 255))),
        contact_edge_floor_applied_px=int(
            np.count_nonzero(masks.contact_edge & (alpha < int(cfg.contact_edge_alpha_floor)))
        ),
        visitor_visible_residual_ratio=(
            float(np.count_nonzero((constrained > 16) & masks.visitor_visible_clear))
            / float(visible_visitor_px)
            if visible_visitor_px
            else 0.0
        ),
        subject_contact_missing_ratio=(
            float(np.count_nonzero((constrained < int(cfg.contact_edge_alpha_floor)) & masks.contact_conflict))
            / float(contact_px)
            if contact_px
            else 0.0
        ),
        body_outside_soft_alpha_ratio=(
            float(np.count_nonzero(body_soft))
            / float(max(np.count_nonzero(foreground & body_region), 1))
        ),
        body_edge_removed_px=int(
            np.count_nonzero((constrained <= int(cfg.hair_refine_min_alpha)) & masks.body_inner_rejudge)
        ),
        body_core_missing_ratio=(
            float(np.count_nonzero((constrained <= int(cfg.hair_refine_min_alpha)) & body_region & core_mask))
            / float(max(np.count_nonzero(body_region & core_mask), 1))
            if body_inner_px
            else 0.0
        ),
    )


class MatAnyoneService:
    def __init__(
        self,
        *,
        model_id: str = "PeiqingYang/MatAnyone2",
        prefer_cuda: bool = True,
        logger: Any | None = None,
    ) -> None:
        self.model_id = model_id
        self.prefer_cuda = prefer_cuda
        self._logger = logger or print

        try:
            from matanyone2 import MatAnyone2, InferenceCore
        except ImportError as exc:
            raise MatAnyoneError(
                "matanyone2 is required. Install with: "
                "pip install -e . (from github.com/pq-yang/MatAnyone2)",
                stage="initialize",
            ) from exc

        self._MatAnyone2 = MatAnyone2
        self.model: Any = MatAnyone2.from_pretrained(model_id)
        device = "cuda:0" if (prefer_cuda and self._cuda_ok()) else "cpu"
        self._inference = InferenceCore(self.model, device=device)
        self._device = device
        self._logger(f"[MatAnyone] model={model_id} device={device}")

    @staticmethod
    def _cuda_ok() -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def process_video(
        self,
        video_path: Path | str,
        first_mask: np.ndarray,
        output_dir: Path | str,
        *,
        save_frames: bool = True,
    ) -> list[Image.Image]:
        video_path = Path(video_path)
        if not video_path.exists():
            raise MatAnyoneError(f"Video not found: {video_path}", stage="read")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        mask_path = output_dir / "first_mask.png"
        if first_mask.dtype == bool:
            mask_img = Image.fromarray(first_mask.astype(np.uint8) * 255, mode="L")
        elif first_mask.dtype == np.uint8 and first_mask.max() <= 1:
            mask_img = Image.fromarray(first_mask * 255, mode="L")
        else:
            mask_img = Image.fromarray(first_mask.astype(np.uint8), mode="L")
        mask_img.save(str(mask_path))

        self._logger(f"[MatAnyone] processing: {video_path}")
        try:
            fgr_path, pha_path = self._inference.process_video(
                input_path=str(video_path),
                mask_path=str(mask_path),
                output_path=str(output_dir),
                save_image=save_frames,
            )
        except Exception as exc:
            raise MatAnyoneError(f"MatAnyone inference failed: {exc}", stage="matting") from exc

        alpha_frames: list[Image.Image] = []
        if save_frames:
            video_stem = video_path.stem
            pha_dir = output_dir / video_stem / "pha"
            if pha_dir.exists():
                for p in sorted(pha_dir.iterdir()):
                    if p.suffix.lower() in (".png", ".jpg"):
                        alpha_frames.append(Image.open(p).convert("L"))

        self._logger(
            f"[MatAnyone] done: {len(alpha_frames)} alpha frames "
            f"pha={pha_path}"
        )
        return alpha_frames
