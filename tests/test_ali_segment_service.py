"""Test ali_segment_service error boundaries."""

from unittest.mock import MagicMock

import pytest

from ali_segment_service import AliSegmentError, AliSegmentService, _wrap_pipeline_init


def test_wrap_pipeline_init_raises_ali_segment_error():
    def factory(**_kwargs):
        raise RuntimeError("boom")

    with pytest.raises(AliSegmentError) as exc_info:
        _wrap_pipeline_init(factory, {}, "Test")

    assert exc_info.value.stage == "initialize"
    assert "Test" in str(exc_info.value)


def test_segment_image_file_wraps_pipeline_error(tmp_path):
    service = AliSegmentService.__new__(AliSegmentService)
    service.provider = "test_provider"
    service.pipeline = MagicMock()
    service.pipeline.process_and_save.side_effect = RuntimeError("network down")

    with pytest.raises(AliSegmentError) as exc_info:
        service.segment_image_file(tmp_path / "source.jpg", tmp_path / "out.png")

    assert exc_info.value.provider == "test_provider"
    assert exc_info.value.stage == "segment"
    assert "network down" in str(exc_info.value)
