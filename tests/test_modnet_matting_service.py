from pathlib import Path
import shutil

import numpy as np
import pytest
from PIL import Image

import modnet_matting_service
from modnet_matting_service import ModnetMattingError, ModnetMattingService, apply_instance_alpha_constraints
from subject_instance_segmentation import InstanceCandidate, InstanceSegmentationConfig, InstanceSegmentationResult, build_instance_trimap


def _case_dir(name: str) -> Path:
    path = Path(".pytest_tmp") / "modnet_matting_service" / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _mask(size=(16, 12), box=(3, 2, 10, 10)):
    arr = np.zeros((size[1], size[0]), dtype=bool)
    x1, y1, x2, y2 = box
    arr[y1:y2, x1:x2] = True
    return arr


def _result() -> InstanceSegmentationResult:
    subject = InstanceCandidate(bbox=(3, 2, 10, 10), confidence=0.9, mask=_mask(), score=0.9)
    visitor = InstanceCandidate(bbox=(11, 2, 15, 9), confidence=0.8, mask=_mask(box=(11, 2, 15, 9)), score=0.4)
    trimap, fg, bg, unknown = build_instance_trimap(
        subject,
        [visitor],
        (16, 12),
        InstanceSegmentationConfig(sure_fg_erode_px=1, subject_unknown_dilate_px=1, visitor_bg_dilate_px=1),
    )
    return InstanceSegmentationResult(
        source_path=Path("capture.jpg"),
        image_size=(16, 12),
        selected=subject,
        candidates=[subject, visitor],
        visitors=[visitor],
        trimap=trimap,
        sure_foreground=fg,
        sure_background=bg,
        unknown=unknown,
    )


class FakeBackend:
    def __init__(self, device_name: str, alpha_value: int = 128) -> None:
        self.device_name = device_name
        self.alpha_value = alpha_value

    def predict_alpha(self, image: Image.Image) -> np.ndarray:
        return np.full((image.height, image.width), self.alpha_value, dtype=np.uint8)


def test_apply_instance_alpha_constraints_forces_foreground_and_background():
    result = _result()
    raw_alpha = np.full((12, 16), 123, dtype=np.uint8)

    constrained = apply_instance_alpha_constraints(Image.new("RGB", (16, 12)), raw_alpha, result)

    assert np.all(constrained.constrained_alpha[result.sure_foreground] == 255)
    assert np.all(constrained.constrained_alpha[result.sure_background] == 0)
    unknown = result.unknown & ~result.sure_foreground & ~result.sure_background
    assert np.all(constrained.constrained_alpha[unknown] == 123)
    assert constrained.image.mode == "RGBA"


def test_modnet_service_falls_back_from_cuda_to_cpu(monkeypatch):
    calls: list[str] = []

    def factory(device: str):
        calls.append(device)
        if device == "cuda":
            raise ModnetMattingError("cuda unavailable", stage="initialize")
        return FakeBackend(device)

    monkeypatch.setattr(modnet_matting_service, "_is_cuda_available", lambda: True)

    service = ModnetMattingService(backend_factory=factory, logger=lambda _msg: None)

    assert service.runtime.device == "cpu"
    assert calls == ["cuda", "cpu"]


def test_matte_image_file_applies_constraints_with_fake_backend():
    case_dir = _case_dir("matte")
    source = case_dir / "capture.jpg"
    Image.new("RGB", (16, 12), (20, 30, 40)).save(source)
    service = ModnetMattingService(backend_factory=lambda device: FakeBackend(device, alpha_value=100), logger=lambda _msg: None)

    instance_result = _result()
    result = service.matte_image_file(source, instance_result)
    alpha = np.array(result.image.getchannel("A"), dtype=np.uint8)

    assert np.all(alpha[instance_result.sure_foreground] == 255)
    assert np.all(alpha[instance_result.sure_background] == 0)
    assert result.forced_foreground_px > 0
    assert result.forced_background_px > 0


def test_matte_image_file_rejects_empty_alpha():
    case_dir = _case_dir("empty")
    source = case_dir / "capture.jpg"
    Image.new("RGB", (16, 12), (20, 30, 40)).save(source)
    service = ModnetMattingService(backend_factory=lambda device: FakeBackend(device, alpha_value=0), logger=lambda _msg: None)

    with pytest.raises(ModnetMattingError, match="empty alpha"):
        service.matte_image_file(source, _result())
