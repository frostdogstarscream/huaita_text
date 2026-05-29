from pathlib import Path

import numpy as np
from PIL import Image

import run_yolo_seg_matting_eval as evaluator
from matanyone_service import MatAnyoneConstraintConfig, apply_matanyone_alpha_constraints
from run_yolo_seg_matting_eval import (
    _save_vitmatte_directional_debug,
    aggregate_modnet_metrics,
    build_instance_config,
    compute_matting_metrics,
    compute_vitmatte_directional_metrics,
    reapply_sure_foreground_alpha,
    restore_constrained_alpha_after_refine,
    trimap_config_to_dict,
)
from subject_instance_segmentation import InstanceCandidate, InstanceSegmentationConfig, InstanceSegmentationResult, build_instance_trimap
from vitmatte_service import VitmatteDirectionalRelaxConfig, build_vitmatte_directional_constraints


def _mask(size=(20, 20), box=(4, 4, 12, 18)):
    arr = np.zeros((size[1], size[0]), dtype=bool)
    x1, y1, x2, y2 = box
    arr[y1:y2, x1:x2] = True
    return arr


def _result() -> InstanceSegmentationResult:
    subject = InstanceCandidate(bbox=(4, 4, 12, 18), confidence=0.9, mask=_mask(), score=0.9)
    visitor = InstanceCandidate(bbox=(14, 4, 19, 16), confidence=0.8, mask=_mask(box=(14, 4, 19, 16)), score=0.4)
    trimap, fg, bg, unknown = build_instance_trimap(
        subject,
        [visitor],
        (20, 20),
        InstanceSegmentationConfig(sure_fg_erode_px=1, subject_unknown_dilate_px=1, visitor_bg_dilate_px=1),
    )
    return InstanceSegmentationResult(
        source_path=Path("capture.jpg"),
        image_size=(20, 20),
        selected=subject,
        candidates=[subject, visitor],
        visitors=[visitor],
        trimap=trimap,
        sure_foreground=fg,
        sure_background=bg,
        unknown=unknown,
    )


def test_compute_matting_metrics_reports_visitor_residual_and_core_missing():
    result = _result()
    image = Image.new("RGBA", (20, 20), (10, 20, 30, 0))
    alpha = np.array(image.getchannel("A"), dtype=np.uint8)
    alpha[result.sure_foreground] = 255
    alpha[result.visitors[0].mask] = 255
    image.putalpha(Image.fromarray(alpha, "L"))

    metrics = compute_matting_metrics(image, result)

    assert metrics["visitor_residual_ratio"] > 0.9
    assert metrics["subject_core_missing_ratio"] == 0
    assert metrics["fragment_count"] >= 1
    assert metrics["foreground_px"] > 0


def test_build_instance_config_only_overrides_requested_fields():
    config = build_instance_config(
        yolo_seg_model_path="models/yolo11x-seg.pt",
        sure_fg_erode_px=6,
    )
    payload = trimap_config_to_dict(config)
    assert payload["sure_fg_erode_px"] == 6
    assert payload["subject_unknown_dilate_px"] == 18
    assert payload["visitor_bg_dilate_px"] == 18


def test_aggregate_modnet_metrics_handles_missing_modnet_frames():
    summary = {"groups": {"g1": {"frames": [{"ok": True}]}}}
    metrics = aggregate_modnet_metrics(summary)
    assert metrics["frame_count"] == 0.0
    assert metrics["subject_core_missing_ratio_avg"] == 1.0


def test_aggregate_comparison_metrics_reports_vitmatte_contact_local_branch():
    frame = {
        "current_aliyun_metrics": {"visitor_residual_ratio": 0.25, "subject_core_missing_ratio": 0.0, "fragment_count": 2, "foreground_px": 90},
        "gpupixel_color_beauty_eval_metrics": {"visitor_residual_ratio": 0.08, "subject_core_missing_ratio": 0.0, "fragment_count": 1, "foreground_px": 95},
        "gpupixel_metrics": {
            "red_cast_score": 4.0,
            "red_cast_delta": -2.5,
            "luminance_gain": 9.5,
            "skin_naturalness_proxy": 11.0,
            "edge_artifact_delta": -1.0,
        },
        "gpupixel_fallback": False,
        "mask_metrics": {"visitor_residual_ratio": 0.1, "subject_core_missing_ratio": 0.0, "fragment_count": 1, "foreground_px": 100},
        "modnet_metrics": {"visitor_residual_ratio": 0.0, "subject_core_missing_ratio": 0.0, "fragment_count": 1, "foreground_px": 110},
        "yolo_seg_aliyun_metrics": {"visitor_residual_ratio": 0.0, "subject_core_missing_ratio": 0.0, "fragment_count": 1, "foreground_px": 108},
        "vitmatte_metrics": {"visitor_residual_ratio": 0.02, "subject_core_missing_ratio": 0.01, "fragment_count": 1, "foreground_px": 106},
        "modelscope_universal_metrics": {"visitor_residual_ratio": 0.03, "subject_core_missing_ratio": 0.0, "fragment_count": 1, "foreground_px": 104},
    }

    metrics = evaluator.aggregate_comparison_metrics(
        {
            "vitmatte_directional_relax_config": {"mode": "contact_local"},
            "groups": {"g1": {"frames": [frame]}},
        }
    )

    assert metrics["current_aliyun"]["visitor_residual_ratio_avg"] == 0.25
    assert metrics["gpupixel_color_beauty_eval"]["visitor_residual_ratio_avg"] == 0.08
    assert metrics["yolo_seg_mask"]["visitor_residual_ratio_avg"] == 0.1
    assert metrics["yolo_seg_modnet"]["visitor_residual_ratio_avg"] == 0.0
    assert metrics["yolo_seg_aliyun"]["visitor_residual_ratio_avg"] == 0.0
    assert metrics["vitmatte_contact_local"]["visitor_residual_ratio_avg"] == 0.02
    assert metrics["modelscope_universal_matting"]["visitor_residual_ratio_avg"] == 0.03
    assert metrics["gpupixel_color_beauty_detail"]["red_cast_delta_avg"] == -2.5
    assert metrics["gpupixel_color_beauty_detail"]["fallback_ratio"] == 0.0


def test_generate_comparison_sheet_places_available_four_branch_rows(tmp_path):
    out_root = tmp_path / "eval"
    group = "test_group"
    for branch, color in (
        ("current_aliyun", (255, 0, 0)),
        ("yolo_seg_mask", (0, 255, 0)),
        ("yolo_seg_modnet", (0, 0, 255)),
        ("yolo_seg_aliyun", (240, 190, 0)),
        ("modelscope_universal_matting", (150, 80, 220)),
    ):
        final_dir = out_root / branch / "final" / group
        final_dir.mkdir(parents=True, exist_ok=True)
        for index in range(1, 5):
            Image.new("RGB", (20, 30), color).save(final_dir / f"{group}_{index}.jpg")

    output_path = out_root / f"{group}_four_way_final_sheet.jpg"
    generated = evaluator.generate_comparison_sheet(out_root=out_root, group=group, kind="final", output_path=output_path)

    assert generated is True
    assert output_path.exists()
    with Image.open(output_path) as sheet:
        assert sheet.width > 4 * 20
        assert sheet.height > 4 * 30


def test_apply_modelscope_universal_constraints_keeps_subject_and_removes_visitors():
    result = _result()
    cutout = Image.new("RGBA", (20, 20), (10, 20, 30, 180))
    alpha = np.array(cutout.getchannel("A"), dtype=np.uint8)
    alpha[result.sure_foreground] = 40
    cutout.putalpha(Image.fromarray(alpha, "L"))

    constrained = evaluator.apply_modelscope_universal_constraints(cutout, result)
    constrained_alpha = np.array(constrained.getchannel("A"), dtype=np.uint8)

    assert np.all(constrained_alpha[result.sure_foreground] == 255)
    assert np.all(constrained_alpha[_mask(box=(14, 4, 19, 16))] == 0)
    assert np.count_nonzero(constrained_alpha[:, :3] > 0) == 0


def test_reapply_sure_foreground_alpha_restores_core_pixels():
    result = _result()
    image = Image.new("RGBA", (20, 20), (10, 20, 30, 0))
    restored = reapply_sure_foreground_alpha(image, result)
    alpha = np.array(restored.getchannel("A"), dtype=np.uint8)
    assert np.all(alpha[result.sure_foreground] == 255)


def test_restore_constrained_alpha_after_refine_preserves_core_and_clears_background():
    result = _result()
    image = Image.new("RGBA", (20, 20), (10, 20, 30, 180))

    restored = restore_constrained_alpha_after_refine(image, result)
    alpha = np.array(restored.getchannel("A"), dtype=np.uint8)

    assert np.all(alpha[result.sure_foreground] == 255)
    assert np.all(alpha[result.sure_background] == 0)


def test_vitmatte_directional_metrics_measure_remaining_core_separately_from_released_pixels():
    result = _result()
    relaxation = build_vitmatte_directional_constraints(
        result,
        VitmatteDirectionalRelaxConfig(mode="side_band", contact_side_erode_px=2, min_vertical_overlap_ratio=0.1),
    )
    image = Image.new("RGBA", (20, 20), (10, 20, 30, 0))
    alpha = np.array(image.getchannel("A"), dtype=np.uint8)
    alpha[relaxation.result.sure_foreground] = 255
    image.putalpha(Image.fromarray(alpha, "L"))

    directional, original = compute_vitmatte_directional_metrics(image, result, relaxation.result)

    assert directional["subject_core_missing_ratio"] == 0
    assert original["subject_core_missing_ratio"] > 0


def test_vitmatte_contact_local_debug_writes_contact_zone_mask(tmp_path):
    result = _result()
    relaxation = build_vitmatte_directional_constraints(
        result,
        VitmatteDirectionalRelaxConfig(
            mode="contact_local",
            contact_search_px=3,
            contact_unknown_depth_px=2,
            contact_vertical_margin_px=0,
            min_vertical_overlap_ratio=0.1,
        ),
    )

    _save_vitmatte_directional_debug(tmp_path, "frame", relaxation)

    assert (tmp_path / "frame_vitmatte_contact_zone.png").exists()
    assert (tmp_path / "frame_vitmatte_directional_unknown.png").exists()


def test_compute_matanyone_metrics_includes_outside_subject_alpha():
    result = _result()
    raw_alpha = np.full((20, 20), 128, dtype=np.uint8)
    constraint = apply_matanyone_alpha_constraints(
        Image.new("RGB", (20, 20), (10, 20, 30)),
        raw_alpha,
        result,
        MatAnyoneConstraintConfig(
            core_erode_px=1,
            body_soft_band_px=1,
            head_soft_band_px=2,
            visitor_clear_dilate_px=0,
        ),
    )

    metrics = evaluator.compute_matanyone_metrics(constraint, result)

    assert metrics["raw_foreground_px"] == 400
    assert metrics["constrained_foreground_px"] > 0
    assert metrics["outside_subject_alpha_ratio"] >= 0
    assert metrics["soft_band_alpha_px"] >= 0
    assert metrics["right_hair_outside_alpha_px"] >= 0
    assert metrics["right_hair_removed_alpha_px"] >= 0
    assert metrics["right_hair_retained_alpha_px"] >= 0
    assert metrics["subject_core_missing_ratio_excluding_hair_rejudge"] == 0


def test_matanyone_debug_writes_right_hair_masks(tmp_path):
    result = _result()
    constraint = apply_matanyone_alpha_constraints(
        Image.new("RGB", (20, 20), (10, 20, 30)),
        np.full((20, 20), 128, dtype=np.uint8),
        result,
        MatAnyoneConstraintConfig(
            core_erode_px=1,
            body_soft_band_px=1,
            head_soft_band_px=2,
            hair_side_refine_enabled=True,
            hair_refine_inner_rejudge_px=1,
            hair_refine_outer_soft_band_px=1,
            visitor_clear_dilate_px=0,
        ),
    )

    evaluator._save_matanyone_debug(tmp_path, "frame", constraint, refined_image=constraint.image)

    assert (tmp_path / "frame_matanyone_right_hair_region.png").exists()
    assert (tmp_path / "frame_matanyone_right_hair_inner_rejudge.png").exists()
    assert (tmp_path / "frame_matanyone_right_hair_outer_support.png").exists()
    assert (tmp_path / "frame_matanyone_right_hair_alpha_before_after.png").exists()


def test_aggregate_matanyone_detail_metrics_counts_every_frame():
    summary = {
        "groups": {
            "g1": {
                "frames": [
                    {"matanyone_metrics": {"outside_subject_alpha_ratio": 0.02, "outside_subject_alpha_px": 4, "subject_core_missing_ratio_excluding_hair_rejudge": 0.0}},
                    {"matanyone_metrics": {"outside_subject_alpha_ratio": 0.04, "outside_subject_alpha_px": 8, "subject_core_missing_ratio_excluding_hair_rejudge": 0.02}},
                ]
            }
        }
    }

    metrics = evaluator.aggregate_matanyone_detail_metrics(summary)

    assert metrics["frame_count"] == 2.0
    assert metrics["outside_subject_alpha_ratio_avg"] == 0.03
    assert metrics["outside_subject_alpha_px_max"] == 8
    assert metrics["subject_core_missing_ratio_excluding_hair_rejudge_avg"] == 0.01


def test_compute_matting_metrics_reports_subject_core_loss():
    result = _result()
    image = Image.new("RGBA", (20, 20), (10, 20, 30, 0))

    metrics = compute_matting_metrics(image, result)

    assert metrics["visitor_residual_ratio"] == 0
    assert metrics["subject_core_missing_ratio"] == 1
    assert metrics["foreground_px"] == 0


def test_compute_matting_metrics_maps_full_image_masks_into_roi_output():
    result = _result()
    roi_image = Image.new("RGBA", (10, 20), (10, 20, 30, 0))
    alpha = np.array(roi_image.getchannel("A"), dtype=np.uint8)
    alpha[4:16, 4:9] = 255
    roi_image.putalpha(Image.fromarray(alpha, "L"))

    metrics = compute_matting_metrics(roi_image, result, roi_box=(10, 0, 20, 20))

    assert metrics["visitor_residual_ratio"] == 1.0
