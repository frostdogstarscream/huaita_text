from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image


class RmbgSegmentError(RuntimeError):
    def __init__(self, message: str, *, stage: str = "segment") -> None:
        super().__init__(message)
        self.stage = stage


class RmbgBackend(Protocol):
    device_name: str

    def segment(self, image: Image.Image) -> Image.Image:
        ...


@dataclass(frozen=True)
class RmbgRuntimeInfo:
    model_id: str
    device: str


def _load_transformers_pipeline() -> Any:
    # Transformers imports sklearn for text-generation helpers, but RMBG image
    # segmentation does not use it. Some kiosk environments contain an
    # unrelated sklearn/pandas stack built against an incompatible NumPy ABI.
    from transformers.utils import import_utils

    import_utils._sklearn_available = False
    from transformers import pipeline

    return pipeline


class TransformersRmbgBackend:
    def __init__(self, *, model_id: str, device: str) -> None:
        try:
            pipeline = _load_transformers_pipeline()
        except ImportError as exc:
            raise RmbgSegmentError(
                f"Failed to import transformers for RMBG-2.0: {exc}",
                stage="initialize",
            ) from exc

        device_index = 0 if device == "cuda" else -1
        self.device_name = device
        try:
            self._pipe = pipeline(
                "image-segmentation",
                model=model_id,
                trust_remote_code=True,
                device=device_index,
            )
        except Exception as exc:
            raise RmbgSegmentError(
                f"Failed to initialize RMBG pipeline on {device}: {exc}",
                stage="initialize",
            ) from exc

    def segment(self, image: Image.Image) -> Image.Image:
        rgb = image.convert("RGB")
        try:
            predictions = self._pipe(rgb)
        except Exception as exc:
            raise RmbgSegmentError(f"RMBG inference failed: {exc}", stage="segment") from exc

        alpha = _extract_alpha_from_predictions(predictions, rgb.size)
        rgba = rgb.convert("RGBA")
        arr = np.array(rgba)
        arr[:, :, 3] = alpha
        return Image.fromarray(arr, "RGBA")


def _extract_alpha_from_predictions(predictions: Any, size: tuple[int, int]) -> np.ndarray:
    if isinstance(predictions, dict):
        predictions = [predictions]
    if not isinstance(predictions, list) or not predictions:
        raise RmbgSegmentError("Unexpected RMBG output format (empty prediction).", stage="decode")

    best_score = -1.0
    best_mask: np.ndarray | None = None
    for item in predictions:
        if not isinstance(item, dict) or "mask" not in item:
            continue
        mask = item["mask"]
        mask_image = mask if isinstance(mask, Image.Image) else Image.fromarray(np.array(mask))
        mask_resized = mask_image.convert("L").resize(size, Image.Resampling.BILINEAR)
        mask_arr = np.array(mask_resized, dtype=np.uint8)
        score = float(item.get("score", 0.0))
        if best_mask is None or score > best_score:
            best_mask = mask_arr
            best_score = score

    if best_mask is None:
        raise RmbgSegmentError("RMBG output missing valid mask.", stage="decode")
    return best_mask


def _is_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


class RmbgSegmentService:
    def __init__(
        self,
        *,
        model_id: str = "briaai/RMBG-2.0",
        prefer_cuda: bool = True,
        backend_factory: Any | None = None,
        logger: Any | None = None,
    ) -> None:
        self.model_id = model_id
        self.prefer_cuda = prefer_cuda
        self._backend_factory = backend_factory or (lambda device: TransformersRmbgBackend(model_id=model_id, device=device))
        self._logger = logger or print
        self.backend = self._initialize_backend()
        self.runtime = RmbgRuntimeInfo(model_id=model_id, device=self.backend.device_name)

    def _initialize_backend(self) -> RmbgBackend:
        if self.prefer_cuda and _is_cuda_available():
            try:
                backend = self._backend_factory("cuda")
                self._logger("[RMBG] device=cuda")
                return backend
            except Exception as exc:
                self._logger(f"[RMBG] cuda init failed, fallback to cpu: {exc}")
        backend = self._backend_factory("cpu")
        self._logger("[RMBG] device=cpu")
        return backend

    def segment_image_file(self, source_path: Path, output_path: Path) -> Image.Image:
        try:
            with Image.open(source_path) as image:
                rgb = image.convert("RGB")
        except Exception as exc:
            raise RmbgSegmentError(f"Failed to read source image: {source_path} ({exc})", stage="read") from exc

        rgba = self.backend.segment(rgb).convert("RGBA")
        alpha = np.array(rgba.getchannel("A"), dtype=np.uint8)
        if int(np.count_nonzero(alpha > 0)) <= 0:
            raise RmbgSegmentError("RMBG produced empty alpha output.", stage="segment")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        rgba.save(output_path, format="PNG")
        return rgba
