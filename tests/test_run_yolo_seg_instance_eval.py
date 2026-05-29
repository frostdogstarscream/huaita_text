from pathlib import Path

import numpy as np
from PIL import Image

from run_yolo_seg_instance_eval import _metrics
from subject_instance_segmentation import (
    InstanceCandidate,
    InstanceSegmentationConfig,
    InstanceSegmentationResult,
    build_instance_trimap,
)


def _mask(size=(100, 100), box=(20, 20, 70, 90)):
    arr = np.zeros((size[1], size[0]), dtype=bool)
    x1, y1, x2, y2 = box
    arr[y1:y2, x1:x2] = True
    return arr


def test_metrics_report_instance_ownership_numbers():
    subject = InstanceCandidate(bbox=(20, 20, 70, 90), confidence=0.9, mask=_mask(box=(20, 20, 70, 90)), score=0.8)
    visitor = InstanceCandidate(bbox=(72, 20, 92, 80), confidence=0.8, mask=_mask(box=(72, 20, 92, 80)), score=0.5)
    trimap, fg, bg, unknown = build_instance_trimap(subject, [visitor], (100, 100), InstanceSegmentationConfig())
    result = InstanceSegmentationResult(
        source_path=Path("capture.jpg"),
        image_size=(100, 100),
        selected=subject,
        candidates=[subject, visitor],
        visitors=[visitor],
        trimap=trimap,
        sure_foreground=fg,
        sure_background=bg,
        unknown=unknown,
    )

    metrics = _metrics(result, 0.12, None)

    assert metrics["ok"] is True
    assert metrics["candidate_count"] == 2
    assert metrics["visitor_count"] == 1
    assert metrics["selected_mask_px"] > 0
    assert metrics["trimap_unknown_px"] > 0


def test_metrics_handles_failed_segmentation():
    metrics = _metrics(None, 0.0, "segmentation failed")

    assert metrics["ok"] is False
    assert metrics["error"] == "segmentation failed"
