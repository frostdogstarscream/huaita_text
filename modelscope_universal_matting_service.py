from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image


MODELSCOPE_UNIVERSAL_MODEL_ID = "iic/cv_unet_universal-matting"


class ModelScopeUniversalMattingError(RuntimeError):
    pass


class ModelScopeUniversalMattingService:
    def __init__(
        self,
        model_id: str = MODELSCOPE_UNIVERSAL_MODEL_ID,
        *,
        pipeline_factory: Callable[[Any, str], Any] | None = None,
    ) -> None:
        self.model_id = model_id
        self._pipeline_factory = pipeline_factory
        self._pipeline = None

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        os.environ.setdefault("MODELSCOPE_CACHE", str((Path("generated") / "modelscope_cache").resolve()))
        if self._pipeline_factory is None:
            try:
                from modelscope.pipelines import pipeline
                from modelscope.utils.constant import Tasks
            except Exception as exc:
                raise ModelScopeUniversalMattingError(f"modelscope_not_available: {exc}") from exc
            self._pipeline_factory = pipeline
            task = Tasks.universal_matting
        else:
            task = "universal_matting"
        try:
            self._pipeline = self._pipeline_factory(task, self.model_id)
        except Exception as exc:
            raise ModelScopeUniversalMattingError(f"pipeline_load_failed: {exc}") from exc
        return self._pipeline

    def warmup(self) -> None:
        self._load_pipeline()

    def segment_image_file(self, source_path: Path | str) -> Image.Image:
        path = Path(source_path)
        with Image.open(path) as source:
            source_rgb = source.convert("RGB")
            source_size = source_rgb.size

        pipeline = self._load_pipeline()
        try:
            raw_result = pipeline(str(path))
        except Exception as exc:
            raise ModelScopeUniversalMattingError(f"inference_failed: {exc}") from exc

        image = self._normalize_output(raw_result, source_rgb)
        if image.size != source_size:
            image = image.resize(source_size, Image.Resampling.BILINEAR)
        alpha = np.array(image.getchannel("A"), dtype=np.uint8)
        if not np.any(alpha > 0):
            raise ModelScopeUniversalMattingError("empty alpha result")
        return image

    def _normalize_output(self, raw_result: Any, source_rgb: Image.Image) -> Image.Image:
        payload = raw_result
        if isinstance(raw_result, dict):
            for key in ("output_img", "output_image", "output"):
                if key in raw_result:
                    payload = raw_result[key]
                    break
            else:
                raise ModelScopeUniversalMattingError("missing output image")

        if isinstance(payload, Image.Image):
            image = payload.convert("RGBA")
        else:
            arr = np.asarray(payload)
            if arr.ndim == 2:
                image = source_rgb.convert("RGBA")
                image.putalpha(Image.fromarray(arr.astype(np.uint8), "L"))
            elif arr.ndim == 3 and arr.shape[2] == 4:
                bgra = arr.astype(np.uint8)
                rgba = bgra[:, :, [2, 1, 0, 3]]
                image = Image.fromarray(rgba, "RGBA")
            elif arr.ndim == 3 and arr.shape[2] == 3:
                bgr = arr.astype(np.uint8)
                rgb = bgr[:, :, [2, 1, 0]]
                image = Image.fromarray(rgb, "RGB").convert("RGBA")
            else:
                raise ModelScopeUniversalMattingError(f"unsupported output shape: {arr.shape}")
        return image.convert("RGBA")
