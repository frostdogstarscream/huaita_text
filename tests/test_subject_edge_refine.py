import numpy as np
from PIL import Image

from subject_edge_refine import (
    SubjectEdgeRefineConfig,
    effective_alpha_bbox,
    refine_subject_edge,
)


def _rgba(size=(100, 100)):
    arr = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    arr[:, :, :3] = 120
    return arr


def test_refine_subject_edge_removes_small_alpha_islands():
    arr = _rgba()
    arr[20:80, 20:80, 3] = 255
    arr[92:95, 5:8, 3] = 255
    image = Image.fromarray(arr, "RGBA")

    result = refine_subject_edge(image, SubjectEdgeRefineConfig(feather_radius_px=0))

    alpha = np.array(result.image.getchannel("A"))
    assert alpha[40, 40] == 255
    assert alpha[93, 6] == 0
    assert result.removed_small_components >= 1


def test_refine_subject_edge_feathers_hard_alpha_boundary():
    arr = _rgba()
    arr[20:80, 20:80, 3] = 255
    image = Image.fromarray(arr, "RGBA")

    result = refine_subject_edge(
        image,
        SubjectEdgeRefineConfig(
            min_component_area_ratio=0,
            open_kernel_px=0,
            feather_radius_px=1.4,
        ),
    )

    alpha = np.array(result.image.getchannel("A"))
    edge_values = alpha[19:22, 50]
    assert any(0 < int(value) < 255 for value in edge_values)


def test_effective_alpha_bbox_ignores_low_alpha_specks():
    arr = _rgba()
    arr[30:70, 30:70, 3] = 255
    arr[2:4, 2:4, 3] = 8
    image = Image.fromarray(arr, "RGBA")

    bbox = effective_alpha_bbox(image, alpha_threshold=16)

    assert bbox == (30, 30, 70, 70)


def test_refine_subject_edge_returns_empty_image_safely():
    image = Image.fromarray(_rgba(), "RGBA")

    result = refine_subject_edge(image, SubjectEdgeRefineConfig())

    assert result.image.tobytes() == image.tobytes()
    assert result.effective_bbox is None


def test_edge_ring_blur_only_changes_ring_area():
    arr = _rgba()
    arr[20:80, 20:80, 3] = 255
    image = Image.fromarray(arr, "RGBA")

    result = refine_subject_edge(
        image,
        SubjectEdgeRefineConfig(
            min_component_area_ratio=0,
            open_kernel_px=0,
            feather_radius_px=0,
            edge_ring_blur_enabled=True,
            edge_ring_inner_px=2,
            edge_ring_outer_px=4,
            edge_ring_sigma=1.2,
        ),
    )
    alpha = np.array(result.image.getchannel("A"))

    assert result.edge_ring_mask is not None
    assert result.edge_ring_blurred_px > 0
    assert result.edge_ring_alpha_delta_mean >= 0
    assert alpha[50, 50] == 255
    assert alpha[0, 0] == 0
    assert np.any((alpha > 0) & (alpha < 255))


def test_disable_edge_ring_blur_keeps_ring_metrics_zero():
    arr = _rgba()
    arr[20:80, 20:80, 3] = 255
    image = Image.fromarray(arr, "RGBA")

    result = refine_subject_edge(
        image,
        SubjectEdgeRefineConfig(
            min_component_area_ratio=0,
            open_kernel_px=0,
            feather_radius_px=0,
            edge_ring_blur_enabled=False,
        ),
    )

    assert result.edge_ring_mask is None
    assert result.edge_ring_blurred_px == 0
    assert result.edge_ring_alpha_delta_mean == 0.0


def test_arm_edge_tighten_generates_mask_without_touching_head_core():
    arr = _rgba((120, 120))
    arr[12:108, 25:95, 3] = 255
    image = Image.fromarray(arr, "RGBA")

    result = refine_subject_edge(
        image,
        SubjectEdgeRefineConfig(
            min_component_area_ratio=0,
            open_kernel_px=0,
            feather_radius_px=0,
            edge_ring_blur_enabled=False,
            arm_edge_tighten_enabled=True,
            arm_edge_tighten_px=1,
        ),
    )
    alpha = np.array(result.image.getchannel("A"))

    assert result.arm_edge_tighten_mask is not None
    assert result.arm_edge_tighten_applied_px > 0
    assert alpha[20, 60] == 255  # head/core remains intact
    assert alpha[95, 25] == 0 or alpha[95, 94] == 0  # lower arm edges can be tightened
