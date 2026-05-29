from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app_state import CUTOUT_DIR, OUTPUT_DIR
from matanyone_service import (
    MatAnyoneConstraintConfig,
    MatAnyoneConstraintResult,
    MatAnyoneService,
    apply_matanyone_alpha_constraints,
)
from run_tracked_video_matting_eval import select_output_frames
from subject_edge_decontam import SubjectEdgeDecontamConfig, decontaminate_subject_edges, save_decontam_debug
from subject_edge_refine import SubjectEdgeRefineConfig, refine_subject_edge
from subject_instance_segmentation import InstanceCandidate, InstanceSegmentationConfig, InstanceSegmentationResult
from subject_instance_tracking import SubjectInstanceTracker, TrackingConfig


class TrackedMattingError(RuntimeError):
    def __init__(self, message: str, *, stage: str = "tracked_matting") -> None:
        super().__init__(message)
        self.stage = stage


@dataclass(frozen=True)
class TrackedMattingConfig:
    enabled: bool = True
    input_frame_count: int = 16
    output_frame_indices: tuple[int, int, int, int] = (3, 7, 10, 13)
    timeout_seconds: float = 20.0
    subject_priority_enabled: bool = True
    debug_enabled: bool = True
    body_refine_policy: str = "aggressive"
    matanyone_constraint: dict[str, Any] | None = None

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> "TrackedMattingConfig":
        data = mapping or {}
        indices = data.get("output_frame_indices", [3, 7, 10, 13])
        try:
            parsed = tuple(int(v) for v in indices)
        except Exception:
            parsed = (3, 7, 10, 13)
        if len(parsed) != 4:
            parsed = (3, 7, 10, 13)
        return cls(
            enabled=bool(data.get("enabled", True)),
            input_frame_count=max(int(data.get("input_frame_count", 16)), 4),
            output_frame_indices=parsed,
            timeout_seconds=max(float(data.get("timeout_seconds", 20.0)), 1.0),
            subject_priority_enabled=bool(data.get("subject_priority_enabled", True)),
            debug_enabled=bool(data.get("debug_enabled", True)),
            body_refine_policy=str(data.get("body_refine_policy", "aggressive")),
            matanyone_constraint=data.get("matanyone_constraint")
            if isinstance(data.get("matanyone_constraint"), dict)
            else None,
        )


def _tracked_frame_to_instance_result(
    frame: Any,
    *,
    source_path: Path,
    image_size: tuple[int, int],
) -> InstanceSegmentationResult:
    if frame.selected is None:
        raise TrackedMattingError("Tracked frame has no selected subject.", stage="tracking")
    selected = InstanceCandidate(
        bbox=frame.selected.bbox,
        confidence=frame.selected.confidence,
        mask=frame.selected.mask,
        score=frame.selected.score,
    )
    visitors = [
        InstanceCandidate(
            bbox=v.bbox,
            confidence=v.confidence,
            mask=v.mask,
            score=v.score,
        )
        for v in frame.visitors
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


def _save_subject_priority_debug(
    constraint: MatAnyoneConstraintResult,
    *,
    output_dir: Path,
    stem: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    masks = constraint.masks
    debug_masks = {
        "contact_conflict": masks.contact_conflict,
        "visitor_visible_clear": masks.visitor_visible_clear,
        "contact_core_restore": masks.contact_core,
        "contact_edge_floor": masks.contact_edge,
        "torso_region": masks.torso_region,
        "arm_region": masks.arm_region,
        "body_inner_rejudge": masks.body_inner_rejudge,
        "body_outer_support": masks.body_outer_support,
    }
    for suffix, mask in debug_masks.items():
        Image.fromarray(mask.astype(np.uint8) * 255, "L").save(
            output_dir / f"{stem}_{suffix}.png",
            format="PNG",
        )
    constraint.image.convert("RGBA").save(output_dir / f"{stem}_subject_after.png", format="PNG")


class TrackedMattingService:
    def __init__(self, config: dict[str, Any]):
        self.provider = "tracked_matanyone"
        self.tracked_config = TrackedMattingConfig.from_mapping(config.get("tracked_matting", {}))
        self.edge_config = SubjectEdgeRefineConfig.from_mapping(config.get("subject_edge_refine", {}))
        self.decontam_config = SubjectEdgeDecontamConfig.from_mapping(config.get("subject_edge_decontam", {}))
        model_path = str(config.get("subject_locator", {}).get("model_path", "models/yolo11x-seg.pt"))
        if not model_path.endswith("-seg.pt"):
            model_path = "models/yolo11x-seg.pt"
        self.tracker = SubjectInstanceTracker(
            instance_config=InstanceSegmentationConfig(model_path=model_path),
            tracking_config=TrackingConfig(model_path=model_path),
        )
        self.matanyone = MatAnyoneService()
        priority_config = MatAnyoneConstraintConfig(
            occlusion_conflict_policy="selected_subject_priority",
            contact_subject_priority_enabled=bool(self.tracked_config.subject_priority_enabled),
            contact_visitor_dilate_px=8,
            contact_core_erode_px=6,
            contact_edge_alpha_floor=140,
            torso_refine_enabled=True,
            torso_inner_rejudge_px=4,
            torso_outer_soft_band_px=1,
            arm_refine_enabled=True,
            arm_inner_rejudge_px=3,
            arm_outer_soft_band_px=1,
            body_contact_edge_alpha_floor=170,
        )
        if str(self.tracked_config.body_refine_policy).lower() == "match_head":
            priority_config = priority_config.with_body_match_head()
            edge_raw = {
                field_name: getattr(self.edge_config, field_name)
                for field_name in SubjectEdgeRefineConfig.__dataclass_fields__
            }
            edge_raw["arm_edge_tighten_enabled"] = False
            self.edge_config = SubjectEdgeRefineConfig.from_mapping(edge_raw)
        overrides = self.tracked_config.matanyone_constraint or {}
        if overrides:
            merged = {
                field_name: getattr(priority_config, field_name)
                for field_name in MatAnyoneConstraintConfig.__dataclass_fields__
            }
            for field_name in MatAnyoneConstraintConfig.__dataclass_fields__:
                if field_name in overrides:
                    merged[field_name] = overrides[field_name]
            priority_config = MatAnyoneConstraintConfig(**merged)
        self.priority_config = priority_config

    def segment_image_file(self, source_path: Path, output_path: Path) -> Image.Image:
        raise TrackedMattingError(
            "tracked_matanyone requires sequence input; single-image API is unsupported.",
            stage="input",
        )

    def segment_sequence(
        self,
        *,
        video_path: Path,
        frame_paths: list[Path],
        output_indices: list[int],
        shot_task_ids: list[str],
        task_id: str,
    ) -> tuple[list[Image.Image], list[str], dict[str, Any]]:
        started = time.perf_counter()
        if len(shot_task_ids) != 4:
            raise TrackedMattingError("shot_task_ids must contain exactly 4 items.", stage="input")
        sequence = self.tracker.track_paths(frame_paths)
        if sequence.status != "ok":
            raise TrackedMattingError(
                f"tracking failed: {sequence.error or sequence.status}",
                stage="tracking",
            )
        selected_frames = select_output_frames(sequence.frames, output_indices)
        alpha_frames = self.matanyone.process_video(
            video_path,
            sequence.frames[0].selected.mask.astype(np.uint8) * 255,  # type: ignore[union-attr]
            OUTPUT_DIR / "subject_debug" / task_id / "raw",
            save_frames=bool(self.tracked_config.debug_enabled),
        )
        selected_alpha = select_output_frames(alpha_frames, output_indices)
        subjects: list[Image.Image] = []
        cutout_urls: list[str] = []
        subject_contact_missing: list[float] = []
        visitor_visible_residual: list[float] = []
        body_outside_soft_alpha: list[float] = []
        body_edge_removed: list[float] = []
        body_core_missing: list[float] = []
        edge_ring_blurred: list[float] = []
        arm_edge_tighten_applied: list[float] = []
        debug_dir = OUTPUT_DIR / "subject_debug" / task_id / "tracked_priority"
        for slot, (source_path, frame, raw_alpha, shot_task_id) in enumerate(
            zip(select_output_frames(frame_paths, output_indices), selected_frames, selected_alpha, shot_task_ids),
            start=1,
        ):
            with Image.open(source_path) as source:
                source_rgb = source.convert("RGB")
                instance = _tracked_frame_to_instance_result(
                    frame,
                    source_path=source_path,
                    image_size=source_rgb.size,
                )
            constrained = apply_matanyone_alpha_constraints(
                source_rgb,
                np.asarray(raw_alpha, dtype=np.uint8),
                instance,
                self.priority_config,
            )
            refine_result = refine_subject_edge(constrained.image, self.edge_config)
            refined = refine_result.image
            restored = apply_matanyone_alpha_constraints(
                refined.convert("RGB"),
                np.asarray(refined.getchannel("A"), dtype=np.uint8),
                instance,
                self.priority_config,
            )
            before_decontam = restored.image
            decontam_result = decontaminate_subject_edges(before_decontam, self.decontam_config)
            final_cutout = decontam_result.image
            cutout_path = CUTOUT_DIR / f"{shot_task_id}.png"
            cutout_path.parent.mkdir(parents=True, exist_ok=True)
            final_cutout.save(cutout_path, format="PNG")
            if self.tracked_config.debug_enabled:
                _save_subject_priority_debug(restored, output_dir=debug_dir, stem=f"{slot:02d}")
                if self.decontam_config.debug_enabled:
                    save_decontam_debug(
                        before_decontam,
                        decontam_result,
                        output_dir=debug_dir,
                        stem=f"{slot:02d}",
                        edge_mask=decontam_result.edge_mask,
                    )
            subject_contact_missing.append(float(restored.subject_contact_missing_ratio))
            visitor_visible_residual.append(float(restored.visitor_visible_residual_ratio))
            body_outside_soft_alpha.append(float(restored.body_outside_soft_alpha_ratio))
            body_edge_removed.append(float(restored.body_edge_removed_px))
            body_core_missing.append(float(restored.body_core_missing_ratio))
            edge_ring_blurred.append(float(refine_result.edge_ring_blurred_px))
            arm_edge_tighten_applied.append(float(refine_result.arm_edge_tighten_applied_px))
            cutout_urls.append(f"/generated/cutouts/{shot_task_id}.png")
            subjects.append(final_cutout)

        metrics = {
            "tracking_status": sequence.status,
            "alpha_frame_count": len(alpha_frames),
            "subject_contact_missing_ratio": float(np.mean(subject_contact_missing)) if subject_contact_missing else 0.0,
            "visitor_visible_residual_ratio": float(np.mean(visitor_visible_residual)) if visitor_visible_residual else 0.0,
            "body_outside_soft_alpha_ratio": (
                float(np.mean(body_outside_soft_alpha)) if body_outside_soft_alpha else 0.0
            ),
            "body_edge_removed_px": float(np.mean(body_edge_removed)) if body_edge_removed else 0.0,
            "body_core_missing_ratio": float(np.mean(body_core_missing)) if body_core_missing else 0.0,
            "edge_ring_blurred_px": float(np.mean(edge_ring_blurred)) if edge_ring_blurred else 0.0,
            "arm_edge_tighten_applied_px": (
                float(np.mean(arm_edge_tighten_applied)) if arm_edge_tighten_applied else 0.0
            ),
            "elapsed_seconds": time.perf_counter() - started,
        }
        return subjects, cutout_urls, metrics
