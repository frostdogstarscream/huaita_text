from pathlib import Path
import shutil
import sys
import types

import numpy as np
import pytest
from PIL import Image

from rmbg_segment_service import RmbgSegmentError, RmbgSegmentService, _load_transformers_pipeline


class _FakeBackend:
    def __init__(self, device_name: str, alpha_value: int = 255):
        self.device_name = device_name
        self._alpha_value = alpha_value

    def segment(self, image: Image.Image) -> Image.Image:
        rgba = image.convert("RGBA")
        arr = np.array(rgba)
        arr[:, :, 3] = self._alpha_value
        return Image.fromarray(arr, "RGBA")


def _case_dir(name: str) -> Path:
    root = Path(".pytest_tmp") / "rmbg_service"
    case = root / name
    if case.exists():
        shutil.rmtree(case)
    case.mkdir(parents=True, exist_ok=True)
    return case


def test_rmbg_service_falls_back_to_cpu_when_cuda_backend_fails(monkeypatch):
    calls: list[str] = []

    def fake_factory(device: str):
        calls.append(device)
        if device == "cuda":
            raise RuntimeError("cuda init failed")
        return _FakeBackend("cpu")

    monkeypatch.setattr("rmbg_segment_service._is_cuda_available", lambda: True)
    service = RmbgSegmentService(backend_factory=fake_factory)

    assert calls == ["cuda", "cpu"]
    assert service.runtime.device == "cpu"


def test_rmbg_service_segments_and_writes_png(monkeypatch):
    case_dir = _case_dir("segment_ok")
    monkeypatch.setattr("rmbg_segment_service._is_cuda_available", lambda: False)
    service = RmbgSegmentService(backend_factory=lambda _device: _FakeBackend("cpu"))

    source = case_dir / "in.jpg"
    output = case_dir / "out.png"
    Image.new("RGB", (20, 20), (10, 20, 30)).save(source, format="JPEG")

    result = service.segment_image_file(source, output)

    assert output.exists()
    assert result.mode == "RGBA"
    assert result.getpixel((10, 10))[3] == 255


def test_rmbg_service_raises_for_empty_alpha(monkeypatch):
    case_dir = _case_dir("segment_empty")
    monkeypatch.setattr("rmbg_segment_service._is_cuda_available", lambda: False)
    service = RmbgSegmentService(backend_factory=lambda _device: _FakeBackend("cpu", alpha_value=0))

    source = case_dir / "in.jpg"
    output = case_dir / "out.png"
    Image.new("RGB", (20, 20), (10, 20, 30)).save(source, format="JPEG")

    with pytest.raises(RmbgSegmentError) as exc_info:
        service.segment_image_file(source, output)

    assert exc_info.value.stage == "segment"


def test_transformers_loader_skips_unused_sklearn_optional_dependency(monkeypatch):
    fake_import_utils = types.SimpleNamespace(_sklearn_available=True)
    fake_pipeline = object()
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.pipeline = fake_pipeline
    fake_utils = types.ModuleType("transformers.utils")
    fake_utils.import_utils = fake_import_utils

    monkeypatch.setitem(sys.modules, "transformers.utils.import_utils", fake_import_utils)
    monkeypatch.setitem(sys.modules, "transformers.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    assert _load_transformers_pipeline() is fake_pipeline
    assert fake_import_utils._sklearn_available is False
