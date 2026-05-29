from pathlib import Path

import numpy as np
from PIL import Image

from subject_alpha_filter import SubjectLocationResult
from subject_locator import SubjectCandidate
from subject_visitor_suppression import (
    SubjectVisitorSuppressionConfig,
    apply_post_alpha_hard_clear,
    build_visitor_suppression_mask,
    suppress_visitors_in_roi,
)


def _location(*, other_bboxes=None) -> SubjectLocationResult:
    return SubjectLocationResult(
        roi_path=Path("roi.jpg"),
        original_size=(100, 100),
        roi_box=(0, 0, 100, 100),
        subject=SubjectCandidate(bbox=(10, 10, 50, 90), confidence=0.9, score=0.8),
        candidates=[],
        other_person_bboxes=list(other_bboxes or []),
    )


def test_suppress_visitors_in_roi_replaces_visitor_region_with_blur(tmp_path):
    roi = Image.new("RGB", (100, 100), (20, 40, 60))
    for y in range(10, 90):
        for x in range(60, 95):
            roi.putpixel((x, y), (250, 250, 250))
    source = tmp_path / "roi.jpg"
    roi.save(source)
    location = _location(other_bboxes=[(60, 10, 95, 90)])

    cleaned = suppress_visitors_in_roi(
        source,
        location,
        output_dir=tmp_path,
        stem="shot",
        config=SubjectVisitorSuppressionConfig(),
    )

    with Image.open(cleaned.cleaned_roi_path) as image:
        assert image.mode == "RGB"
        assert image.getpixel((80, 50)) != (250, 250, 250)


def test_suppress_visitors_in_roi_does_not_cover_subject_core(tmp_path):
    roi = Image.new("RGB", (100, 100), (20, 40, 60))
    for y in range(46, 55):
        for x in range(41, 50):
            roi.putpixel((x, y), (10, 200, 10))
    source = tmp_path / "roi.jpg"
    roi.save(source)
    location = _location(other_bboxes=[(35, 20, 80, 90)])

    cleaned = suppress_visitors_in_roi(
        source,
        location,
        output_dir=tmp_path,
        stem="shot",
        config=SubjectVisitorSuppressionConfig(fill_mode="solid_background"),
    )

    with Image.open(cleaned.cleaned_roi_path) as image:
        protected = image.getpixel((45, 50))
        assert protected[1] > 140
        cleared = image.getpixel((70, 50))
        assert all(abs(value - expected) <= 2 for value, expected in zip(cleared, (20, 40, 60)))


def test_suppress_visitors_in_roi_returns_original_when_no_visitors(tmp_path):
    roi = Image.new("RGB", (100, 100), (20, 40, 60))
    source = tmp_path / "roi.jpg"
    roi.save(source)

    cleaned = suppress_visitors_in_roi(
        source,
        _location(),
        output_dir=tmp_path,
        stem="shot",
        config=SubjectVisitorSuppressionConfig(),
    )

    assert cleaned.cleaned_roi_path == source
    assert cleaned.visitor_mask_pixels == 0


def test_post_alpha_hard_clear_removes_connected_visitor_area():
    rgba = np.zeros((100, 100, 4), dtype=np.uint8)
    rgba[:, :, :3] = 120
    rgba[10:90, 10:95, 3] = 255
    image = Image.fromarray(rgba, "RGBA")
    location = _location(other_bboxes=[(60, 10, 95, 90)])

    result = apply_post_alpha_hard_clear(
        image,
        location,
        SubjectVisitorSuppressionConfig(),
    )

    alpha = np.array(result.getchannel("A"))
    assert alpha[50, 35] == 255
    assert alpha[50, 75] == 0


def test_small_subject_protect_ratio_does_not_protect_right_side_visitor():
    location = _location(other_bboxes=[(48, 20, 95, 90)])
    mask, _boxes, protect_box = build_visitor_suppression_mask(
        location,
        (100, 100),
        SubjectVisitorSuppressionConfig(
            visitor_preclean_expand_ratio=0.18,
            subject_protect_expand_ratio=0.04,
        ),
    )

    assert protect_box == (8, 7, 52, 93)
    assert mask[50, 70] is np.True_
    assert mask[50, 45] is np.False_


def test_suppress_visitors_in_roi_inpaint_modifies_visitor_region(tmp_path):
    roi = Image.new("RGB", (100, 100), (20, 40, 60))
    for y in range(10, 90):
        for x in range(60, 95):
            roi.putpixel((x, y), (250, 250, 250))
    source = tmp_path / "roi.jpg"
    roi.save(source)
    location = _location(other_bboxes=[(60, 10, 95, 90)])

    cleaned = suppress_visitors_in_roi(
        source,
        location,
        output_dir=tmp_path,
        stem="shot",
        config=SubjectVisitorSuppressionConfig(fill_mode="inpaint", inpaint_radius=7),
    )

    with Image.open(cleaned.cleaned_roi_path) as image:
        assert image.mode == "RGB"
        assert image.getpixel((80, 50)) != (250, 250, 250)


def test_suppress_visitors_in_roi_inpaint_preserves_subject_core(tmp_path):
    roi = Image.new("RGB", (100, 100), (20, 40, 60))
    for y in range(46, 55):
        for x in range(41, 50):
            roi.putpixel((x, y), (10, 200, 10))
    source = tmp_path / "roi.jpg"
    roi.save(source)
    location = _location(other_bboxes=[(35, 20, 80, 90)])

    cleaned = suppress_visitors_in_roi(
        source,
        location,
        output_dir=tmp_path,
        stem="shot",
        config=SubjectVisitorSuppressionConfig(
            fill_mode="inpaint", inpaint_radius=7, subject_protect_expand_ratio=0.15,
        ),
    )

    with Image.open(cleaned.cleaned_roi_path) as image:
        protected = image.getpixel((45, 50))
        assert protected[1] > 140
