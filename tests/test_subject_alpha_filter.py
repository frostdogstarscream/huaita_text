from pathlib import Path

import numpy as np
from PIL import Image

from subject_alpha_filter import (
    SubjectAlphaFilterConfig,
    SubjectLocationResult,
    filter_subject_alpha,
    map_bbox_to_output,
)
from subject_locator import SubjectCandidate


def _location(
    *,
    roi_box=(50, 20, 250, 220),
    subject_bbox=(90, 60, 170, 180),
    other_bboxes=None,
) -> SubjectLocationResult:
    return SubjectLocationResult(
        roi_path=Path("roi.jpg"),
        original_size=(300, 260),
        roi_box=roi_box,
        subject=SubjectCandidate(bbox=subject_bbox, confidence=0.9, score=0.8),
        candidates=[SubjectCandidate(bbox=subject_bbox, confidence=0.9, score=0.8)],
        other_person_bboxes=list(other_bboxes or []),
    )


def _rgba_with_alpha(size, boxes):
    rgba = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    rgba[:, :, :3] = 120
    for left, top, right, bottom in boxes:
        rgba[top:bottom, left:right, 3] = 255
    return Image.fromarray(rgba, "RGBA")


def test_map_bbox_to_output_converts_original_bbox_through_roi_and_result_scale():
    bbox = map_bbox_to_output(
        (90, 60, 170, 180),
        roi_box=(50, 20, 250, 220),
        output_size=(100, 100),
    )

    assert bbox == (20, 20, 60, 80)


def test_filter_subject_alpha_keeps_component_overlapping_subject_box():
    image = _rgba_with_alpha((120, 100), [(22, 25, 58, 75), (92, 20, 114, 55)])
    # Disable morph bridge so the two components stay separate
    result = filter_subject_alpha(image, _location(), SubjectAlphaFilterConfig(morph_close_kernel_px=0))

    alpha = np.array(result.getchannel("A"))
    assert alpha[40, 35] == 255
    assert alpha[30, 100] == 0


def test_filter_subject_alpha_removes_visitor_box_without_erasing_subject_overlap():
    image = _rgba_with_alpha((100, 100), [(20, 20, 62, 82), (62, 22, 92, 80)])
    location = _location(other_bboxes=[(174, 64, 236, 180)])

    result = filter_subject_alpha(image, location, SubjectAlphaFilterConfig())

    alpha = np.array(result.getchannel("A"))
    assert alpha[50, 50] == 255
    assert alpha[50, 82] == 0


def test_filter_subject_alpha_returns_original_when_alpha_is_empty():
    image = Image.new("RGBA", (100, 100), (1, 2, 3, 0))

    result = filter_subject_alpha(image, _location(), SubjectAlphaFilterConfig())

    assert result.tobytes() == image.tobytes()


def test_filter_subject_alpha_morph_bridge_keeps_disconnected_parts():
    # Two subject components 15px apart — bridged by morph_close_kernel_px=20
    rgba = np.zeros((100, 120, 4), dtype=np.uint8)
    rgba[:, :, :3] = 120
    rgba[30:50, 20:45, 3] = 255   # component A (inside subject box)
    rgba[30:50, 60:80, 3] = 255   # component B (15px gap, inside subject box)
    image = Image.fromarray(rgba, "RGBA")
    config = SubjectAlphaFilterConfig(morph_close_kernel_px=20)

    result = filter_subject_alpha(image, _location(), config)

    alpha = np.array(result.getchannel("A"))
    assert alpha[40, 30] == 255, "component A should be kept"
    assert alpha[40, 70] == 255, "component B should be bridged and kept"


def test_filter_subject_alpha_morph_bridge_disabled_with_zero_kernel():
    # Same layout but morph_close_kernel_px=0 — component B should be dropped
    rgba = np.zeros((100, 120, 4), dtype=np.uint8)
    rgba[:, :, :3] = 120
    rgba[30:50, 20:45, 3] = 255
    rgba[30:50, 60:80, 3] = 255
    image = Image.fromarray(rgba, "RGBA")
    config = SubjectAlphaFilterConfig(morph_close_kernel_px=0, keep_nearby_component_px=0)

    result = filter_subject_alpha(image, _location(), config)

    alpha = np.array(result.getchannel("A"))
    assert alpha[40, 30] == 255, "component A should be kept"
    assert alpha[40, 70] == 0, "component B should be dropped without bridging"


def test_filter_subject_alpha_morph_bridge_respects_visitor_mask():
    # Visitor far from subject — visitor pixels inside visitor mask are zeroed
    rgba = np.zeros((100, 120, 4), dtype=np.uint8)
    rgba[:, :, :3] = 120
    rgba[30:50, 20:45, 3] = 255   # subject component
    rgba[60:80, 90:110, 3] = 255  # visitor component (far from subject)
    image = Image.fromarray(rgba, "RGBA")
    # Visitor box maps to image region covering (84,60)-(120,80)
    location = _location(other_bboxes=[(190, 140, 250, 180)])
    config = SubjectAlphaFilterConfig(morph_close_kernel_px=20, visitor_box_expand_ratio=0.0)

    result = filter_subject_alpha(image, location, config)

    alpha = np.array(result.getchannel("A"))
    assert alpha[40, 30] == 255, "subject should be kept"
    assert alpha[70, 100] == 0, "visitor pixels in visitor mask should be zeroed"
