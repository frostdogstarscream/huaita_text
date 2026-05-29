import numpy as np
from PIL import Image
from pathlib import Path

from run_rmbg_ab_eval import _compute_metrics
from subject_alpha_filter import SubjectLocationResult
from subject_locator import SubjectCandidate
from subject_visitor_suppression import SubjectVisitorSuppressionConfig


def _make_cutout() -> Image.Image:
    arr = np.zeros((100, 100, 4), dtype=np.uint8)
    arr[:, :, :3] = 120
    arr[20:80, 20:60, 3] = 255
    arr[20:60, 70:90, 3] = 255  # visitor-like residual
    return Image.fromarray(arr, "RGBA")


def test_compute_metrics_without_location():
    image = _make_cutout()
    metrics = _compute_metrics(image, None, SubjectVisitorSuppressionConfig())

    assert metrics["foreground_px"] > 0
    assert metrics["visitor_residual_ratio"] == 0.0
    assert metrics["fragment_count"] >= 1


def test_compute_metrics_with_visitor_location():
    image = _make_cutout()
    location = SubjectLocationResult(
        roi_path=Path("dummy.jpg"),
        original_size=(100, 100),
        roi_box=(0, 0, 100, 100),
        subject=SubjectCandidate(bbox=(20, 20, 60, 80), confidence=0.9),
        candidates=[],
        other_person_bboxes=[(70, 20, 90, 60)],
    )
    metrics = _compute_metrics(image, location, SubjectVisitorSuppressionConfig())

    assert metrics["visitor_residual_ratio"] > 0.0
    assert metrics["right_top_risk_ratio"] >= 0.0
