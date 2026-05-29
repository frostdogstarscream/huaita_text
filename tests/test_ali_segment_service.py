"""Tests for ali_segment_service with online YOLO-seg primary/visitor logic."""

from pathlib import Path
import threading
from unittest.mock import MagicMock

import numpy as np
from PIL import Image
import pytest

from ali_segment_service import AliSegmentError, AliSegmentService, _wrap_pipeline_init
from subject_alpha_filter import SubjectAlphaFilterConfig
from subject_edge_refine import SubjectEdgeRefineConfig
from subject_instance_segmentation import (
    InstanceCandidate,
    InstanceSegmentationResult,
)
from subject_visitor_suppression import SubjectVisitorSuppressionConfig


def test_wrap_pipeline_init_raises_ali_segment_error():
    def factory(**_kwargs):
        raise RuntimeError("boom")

    with pytest.raises(AliSegmentError) as exc_info:
        _wrap_pipeline_init(factory, {}, "Test")

    assert exc_info.value.stage == "initialize"
    assert "Test" in str(exc_info.value)


def test_segment_image_file_wraps_pipeline_error_after_instance_stage(tmp_path):
    source = tmp_path / "capture.jpg"
    output = tmp_path / "out.png"
    Image.new("RGB", (100, 100), (10, 20, 30)).save(source)

    service = AliSegmentService.__new__(AliSegmentService)
    service.provider = "test_provider"
    service.pipeline = MagicMock()
    service.pipeline.process_and_save.side_effect = RuntimeError("network down")
    service.instance_segmenter = MagicMock()
    service.instance_segmenter.segment.return_value = _instance_result()
    service.alpha_filter_config = SubjectAlphaFilterConfig(enabled=False)
    service.visitor_suppression_config = SubjectVisitorSuppressionConfig(enabled=False)
    service.edge_refine_config = SubjectEdgeRefineConfig(enabled=False)

    with pytest.raises(AliSegmentError) as exc_info:
        service.segment_image_file(source, output)

    assert exc_info.value.provider == "test_provider"
    assert exc_info.value.stage == "segment"
    assert "network down" in str(exc_info.value)


def test_segment_image_file_uses_instance_masks_for_visitor_cleanup(tmp_path):
    source = tmp_path / "capture.jpg"
    output = tmp_path / "out.png"
    Image.new("RGB", (100, 100), (120, 120, 120)).save(source)

    service = AliSegmentService.__new__(AliSegmentService)
    service.provider = "test_provider"
    service.pipeline = MagicMock()
    service.pipeline.process_and_save.return_value = _bgra_two_people_result()
    service.instance_segmenter = MagicMock()
    service.instance_segmenter.segment.return_value = _instance_result()
    service.alpha_filter_config = SubjectAlphaFilterConfig()
    service.visitor_suppression_config = SubjectVisitorSuppressionConfig()
    service.edge_refine_config = SubjectEdgeRefineConfig(enabled=False)

    image = service.segment_image_file(source, output)

    assert image.getpixel((25, 40))[3] == 255
    assert image.getpixel((75, 40))[3] == 0


def test_segment_image_file_fails_when_no_instance_selected(tmp_path):
    source = tmp_path / "capture.jpg"
    output = tmp_path / "out.png"
    Image.new("RGB", (32, 32), (10, 10, 10)).save(source)

    service = AliSegmentService.__new__(AliSegmentService)
    service.provider = "modelscope_universal"
    service.modelscope_universal = MagicMock()
    service.modelscope_universal.segment_image_file.return_value = Image.new("RGBA", (32, 32), (1, 2, 3, 255))
    service.instance_segmenter = MagicMock()
    service.instance_segmenter.segment.return_value = None
    service.alpha_filter_config = SubjectAlphaFilterConfig(enabled=False)
    service.visitor_suppression_config = SubjectVisitorSuppressionConfig(enabled=False)
    service.edge_refine_config = SubjectEdgeRefineConfig(enabled=False)

    with pytest.raises(AliSegmentError, match="instance_segmentation_failed"):
        service.segment_image_file(source, output)


def test_segment_image_file_uses_modelscope_provider_and_emits_metrics(tmp_path):
    source = tmp_path / "capture.jpg"
    output = tmp_path / "out.png"
    Image.new("RGB", (16, 12), (20, 30, 40)).save(source)

    class _FakeModelScope:
        def __init__(self):
            self.calls = []

        def segment_image_file(self, path):
            self.calls.append(path)
            return Image.new("RGBA", (16, 12), (100, 90, 80, 255))

    service = AliSegmentService.__new__(AliSegmentService)
    service.provider = "modelscope_universal"
    service.modelscope_universal = _FakeModelScope()
    service.instance_segmenter = MagicMock()
    service.instance_segmenter.segment.return_value = _instance_result(size=(16, 12))
    service.alpha_filter_config = SubjectAlphaFilterConfig(enabled=False)
    service.visitor_suppression_config = SubjectVisitorSuppressionConfig(enabled=False)
    service.edge_refine_config = SubjectEdgeRefineConfig(enabled=False)
    service._instance_metrics_lock = threading.Lock()
    service._instance_metrics_by_stem = {}

    image = service.segment_image_file(source, output)

    assert image.mode == "RGBA"
    assert image.getpixel((0, 0)) == (100, 90, 80, 255)
    assert output.exists()
    assert service.get_instance_metrics(output.stem)["visitors_count"] == 1


def _instance_result(size: tuple[int, int] = (100, 100)) -> InstanceSegmentationResult:
    width, height = size
    subject_mask = np.zeros((height, width), dtype=bool)
    subject_mask[10:80, 10:45] = True
    visitor_mask = np.zeros((height, width), dtype=bool)
    visitor_mask[10:80, 60:95] = True
    selected = InstanceCandidate(bbox=(10.0, 10.0, 45.0, 80.0), confidence=0.9, mask=subject_mask, score=0.8)
    visitor = InstanceCandidate(bbox=(60.0, 10.0, 95.0, 80.0), confidence=0.8, mask=visitor_mask, score=0.5)
    return InstanceSegmentationResult(
        source_path=Path("."),
        image_size=(width, height),
        selected=selected,
        candidates=[selected, visitor],
        visitors=[visitor],
        trimap=np.zeros((height, width), dtype=np.uint8),
        sure_foreground=subject_mask.copy(),
        sure_background=np.logical_not(subject_mask),
        unknown=np.zeros((height, width), dtype=bool),
    )


def _bgra_two_people_result():
    arr = np.zeros((100, 100, 4), dtype=np.uint8)
    arr[:, :, :3] = 120
    arr[10:80, 10:45, 3] = 255
    arr[10:80, 60:95, 3] = 255
    return arr
