from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from subject_instance_segmentation import (
    InstanceCandidate,
    InstanceSegmentationConfig,
    SubjectInstanceSegmenter,
    build_instance_trimap,
    choose_primary_instance,
    cutout_from_instance_mask,
)


def _mask(size=(100, 100), box=(20, 20, 70, 90)):
    arr = np.zeros((size[1], size[0]), dtype=bool)
    x1, y1, x2, y2 = box
    arr[y1:y2, x1:x2] = True
    return arr


def _case_dir(name: str) -> Path:
    path = Path(".pytest_tmp") / "subject_instance_segmentation" / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_choose_primary_instance_prefers_front_center_person():
    cfg = InstanceSegmentationConfig()
    back = InstanceCandidate(bbox=(45, 10, 70, 55), confidence=0.9, mask=_mask(box=(45, 10, 70, 55)))
    front = InstanceCandidate(bbox=(20, 25, 80, 98), confidence=0.8, mask=_mask(box=(20, 25, 80, 98)))

    selected = choose_primary_instance([back, front], (100, 100), cfg)

    assert selected is not None
    assert selected.bbox == front.bbox
    assert selected.score is not None


def test_build_instance_trimap_marks_visitors_as_sure_background():
    cfg = InstanceSegmentationConfig(
        sure_fg_erode_px=2,
        subject_unknown_dilate_px=4,
        visitor_bg_dilate_px=2,
    )
    subject = InstanceCandidate(bbox=(20, 20, 65, 90), confidence=0.9, mask=_mask(box=(20, 20, 65, 90)))
    visitor = InstanceCandidate(bbox=(68, 25, 92, 80), confidence=0.9, mask=_mask(box=(68, 25, 92, 80)))

    trimap, sure_fg, sure_bg, unknown = build_instance_trimap(subject, [visitor], (100, 100), cfg)

    assert trimap.shape == (100, 100)
    assert sure_fg[40, 40]
    assert sure_bg[40, 80]
    assert not unknown[40, 80]
    assert set(np.unique(trimap)).issubset({0, 128, 255})


def test_subject_instance_segmenter_uses_injected_detector_and_saves_debug():
    case_dir = _case_dir("segmenter_debug")
    source = case_dir / "capture.jpg"
    Image.new("RGB", (100, 100), (10, 20, 30)).save(source)
    subject = InstanceCandidate(bbox=(20, 20, 65, 90), confidence=0.9, mask=_mask(box=(20, 20, 65, 90)))
    visitor = InstanceCandidate(bbox=(68, 25, 92, 80), confidence=0.9, mask=_mask(box=(68, 25, 92, 80)))

    class FakeDetector:
        def detect(self, _source_path: Path, _image_size: tuple[int, int]):
            return [visitor, subject]

    segmenter = SubjectInstanceSegmenter(InstanceSegmentationConfig(), detector=FakeDetector(), output_dir=case_dir / "debug")
    result = segmenter.segment(source, "shot_1")

    assert result is not None
    assert result.selected.bbox == subject.bbox
    assert len(result.visitors) == 1
    assert (case_dir / "debug" / "shot_1_seg_instance_debug.jpg").exists()
    assert (case_dir / "debug" / "shot_1_trimap.png").exists()


def test_cutout_from_instance_mask_uses_selected_mask_as_alpha():
    case_dir = _case_dir("cutout")
    source = case_dir / "capture.jpg"
    Image.new("RGB", (100, 100), (10, 20, 30)).save(source)
    candidate = InstanceCandidate(bbox=(20, 20, 65, 90), confidence=0.9, mask=_mask(box=(20, 20, 65, 90)))

    cutout = cutout_from_instance_mask(source, candidate)

    assert cutout.mode == "RGBA"
    assert cutout.getpixel((40, 40))[3] > 0
    assert cutout.getpixel((5, 5))[3] == 0
