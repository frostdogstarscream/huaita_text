import os
from pathlib import Path

from PIL import Image

from subject_locator import (
    SubjectCandidate,
    SubjectLocator,
    SubjectLocatorConfig,
    choose_primary_subject,
    crop_expanded_roi,
    expand_bbox,
    trim_roi_away_from_side_visitors,
)


def test_choose_primary_subject_selects_single_person():
    cfg = SubjectLocatorConfig()
    candidate = SubjectCandidate(bbox=(100, 80, 420, 700), confidence=0.82)

    selected = choose_primary_subject([candidate], (720, 960), cfg)

    assert selected is not None
    assert selected.bbox == candidate.bbox


def test_choose_primary_subject_prefers_front_center_subject_over_back_tourist():
    cfg = SubjectLocatorConfig()
    back_tourist = SubjectCandidate(bbox=(260, 80, 430, 360), confidence=0.91)
    front_subject = SubjectCandidate(bbox=(170, 220, 570, 930), confidence=0.80)

    selected = choose_primary_subject([back_tourist, front_subject], (720, 960), cfg)

    assert selected is not None
    assert selected.bbox == front_subject.bbox
    assert selected.score is not None
    assert selected.score > 0.7


def test_choose_primary_subject_rejects_empty_low_confidence_and_too_small_people():
    cfg = SubjectLocatorConfig(min_confidence=0.45, min_person_height_ratio=0.25)
    low_conf = SubjectCandidate(bbox=(100, 100, 400, 820), confidence=0.20)
    too_small = SubjectCandidate(bbox=(280, 100, 360, 240), confidence=0.95)

    assert choose_primary_subject([], (720, 960), cfg) is None
    assert choose_primary_subject([low_conf], (720, 960), cfg) is None
    assert choose_primary_subject([too_small], (720, 960), cfg) is None


def test_expand_bbox_clamps_to_image_bounds():
    expanded = expand_bbox((10, 20, 100, 180), (120, 200), 0.30)

    assert expanded == (0, 0, 120, 200)


def test_crop_expanded_roi_saves_rgb_jpg_without_overflow(tmp_path):
    source = tmp_path / "capture.jpg"
    Image.new("RGB", (120, 200), (10, 20, 30)).save(source)

    roi_path = crop_expanded_roi(
        source,
        (10, 20, 100, 180),
        output_dir=tmp_path / "rois",
        stem="shot_1",
        expand_ratio=0.30,
    )

    assert roi_path.exists()
    assert roi_path.suffix.lower() == ".jpg"
    with Image.open(roi_path) as saved:
        assert saved.mode == "RGB"
        assert saved.size == (120, 200)


def test_subject_locator_returns_roi_path_when_detector_finds_subject(tmp_path):
    source = tmp_path / "capture.jpg"
    Image.new("RGB", (720, 960), (10, 20, 30)).save(source)

    class FakeDetector:
        def detect(self, _source_path: Path):
            return [SubjectCandidate(bbox=(170, 220, 570, 930), confidence=0.80)]

    locator = SubjectLocator(SubjectLocatorConfig(), detector=FakeDetector(), output_dir=tmp_path / "rois")
    location = locator.locate(source, "shot_1")

    assert location is not None
    assert location.roi_path.exists()
    assert location.roi_path.parent == tmp_path / "rois"
    assert location.original_size == (720, 960)
    assert location.subject.bbox == (170, 220, 570, 930)
    assert location.roi_box == (122, 135, 618, 960)


def test_subject_locator_reports_other_people_inside_roi(tmp_path):
    source = tmp_path / "capture.jpg"
    Image.new("RGB", (720, 960), (10, 20, 30)).save(source)

    front = SubjectCandidate(bbox=(170, 220, 570, 930), confidence=0.80)
    back = SubjectCandidate(bbox=(250, 80, 430, 360), confidence=0.91)
    outside = SubjectCandidate(bbox=(0, 20, 50, 300), confidence=0.88)

    class FakeDetector:
        def detect(self, _source_path: Path):
            return [back, front, outside]

    locator = SubjectLocator(SubjectLocatorConfig(), detector=FakeDetector(), output_dir=tmp_path / "rois")
    location = locator.locate(source, "shot_1")

    assert location is not None
    assert back.bbox in location.other_person_bboxes
    assert outside.bbox not in location.other_person_bboxes


def test_trim_roi_away_from_right_side_visitor_when_overlap_is_low():
    cfg = SubjectLocatorConfig(
        roi_expand_ratio=0.30,
        roi_side_trim_enabled=True,
        roi_side_trim_margin_ratio=0.08,
        roi_side_trim_max_overlap_ratio=0.20,
    )
    subject = SubjectCandidate(bbox=(100, 100, 300, 500), confidence=0.9)
    visitor = SubjectCandidate(bbox=(310, 140, 430, 500), confidence=0.8)
    roi = expand_bbox(subject.bbox, (520, 620), cfg.roi_expand_ratio)

    trimmed, info = trim_roi_away_from_side_visitors(roi, subject, [visitor], (520, 620), cfg)

    assert trimmed[2] < roi[2]
    assert trimmed[2] == 316
    assert info == "right"


def test_trim_roi_keeps_base_roi_when_visitor_heavily_overlaps_subject():
    cfg = SubjectLocatorConfig(
        roi_expand_ratio=0.30,
        roi_side_trim_enabled=True,
        roi_side_trim_margin_ratio=0.08,
        roi_side_trim_max_overlap_ratio=0.20,
    )
    subject = SubjectCandidate(bbox=(100, 100, 300, 500), confidence=0.9)
    visitor = SubjectCandidate(bbox=(220, 130, 390, 500), confidence=0.8)
    roi = expand_bbox(subject.bbox, (520, 620), cfg.roi_expand_ratio)

    trimmed, info = trim_roi_away_from_side_visitors(roi, subject, [visitor], (520, 620), cfg)

    assert trimmed == roi
    assert info == ""


def test_subject_locator_returns_none_when_disabled_or_detector_fails(tmp_path):
    source = tmp_path / "capture.jpg"
    Image.new("RGB", (720, 960), (10, 20, 30)).save(source)

    class BrokenDetector:
        def detect(self, _source_path: Path):
            raise RuntimeError("model missing")

    disabled = SubjectLocator(SubjectLocatorConfig(enabled=False), detector=BrokenDetector())
    failing = SubjectLocator(SubjectLocatorConfig(), detector=BrokenDetector())

    assert disabled.locate(source, "shot_1") is None
    assert failing.locate(source, "shot_1") is None


def test_subject_locator_sets_yolo_config_dir_under_generated_output(tmp_path, monkeypatch):
    monkeypatch.delenv("YOLO_CONFIG_DIR", raising=False)

    SubjectLocator(SubjectLocatorConfig(), detector=None, output_dir=tmp_path / "subject_rois")

    assert Path(os.environ["YOLO_CONFIG_DIR"]) == tmp_path / "ultralytics"
