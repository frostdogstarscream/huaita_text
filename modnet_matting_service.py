from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from PIL import Image

from subject_instance_segmentation import InstanceSegmentationResult


class ModnetMattingError(RuntimeError):
    def __init__(self, message: str, *, stage: str = "matting") -> None:
        super().__init__(message)
        self.stage = stage


class ModnetBackend(Protocol):
    device_name: str

    def predict_alpha(self, image: Image.Image) -> np.ndarray:
        ...


@dataclass(frozen=True)
class ModnetRuntimeInfo:
    device: str
    repo_path: str
    checkpoint_path: str


@dataclass(frozen=True)
class AlphaConstraintResult:
    image: Image.Image
    raw_alpha: np.ndarray
    constrained_alpha: np.ndarray
    forced_foreground_px: int
    forced_background_px: int
    unknown_px: int


def _is_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


class LocalModnetBackend:
    def __init__(self, *, repo_path: Path, checkpoint_path: Path, device: str) -> None:
        self.device_name = device
        self._repo_path = repo_path
        self._checkpoint_path = checkpoint_path
        if not repo_path.exists():
            raise ModnetMattingError(f"MODNet repo not found: {repo_path}", stage="initialize")
        if not checkpoint_path.exists():
            raise ModnetMattingError(f"MODNet checkpoint not found: {checkpoint_path}", stage="initialize")

        try:
            import torch
        except ImportError as exc:
            raise ModnetMattingError("torch is required for MODNet inference.", stage="initialize") from exc

        sys.path.insert(0, str(repo_path))
        try:
            modnet_module = importlib.import_module("src.models.modnet")
            modnet_cls = getattr(modnet_module, "MODNet")
            model = modnet_cls(backbone_pretrained=False)
            state = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            # Strip "module." prefix from DataParallel/DistributedDataParallel checkpoints
            state = {k.removeprefix("module."): v for k, v in state.items()}
            model.load_state_dict(state, strict=False)
            model.to(device)
            model.eval()
        except Exception as exc:
            raise ModnetMattingError(f"Failed to initialize MODNet: {exc}", stage="initialize") from exc

        self._torch = torch
        self._model = model

    def predict_alpha(self, image: Image.Image) -> np.ndarray:
        torch = self._torch
        rgb = image.convert("RGB")
        width, height = rgb.size
        target_w = max(32, int(np.ceil(width / 32.0) * 32))
        target_h = max(32, int(np.ceil(height / 32.0) * 32))
        resized = rgb.resize((target_w, target_h), Image.Resampling.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(self.device_name)
        try:
            with torch.no_grad():
                output = self._model(tensor, True)
                matte = output[-1] if isinstance(output, (tuple, list)) else output
            alpha = matte[0][0].detach().cpu().numpy()
        except Exception as exc:
            raise ModnetMattingError(f"MODNet inference failed: {exc}", stage="matting") from exc
        alpha = cv2.resize(alpha.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
        return np.clip(alpha * 255.0, 0, 255).astype(np.uint8)


class ModnetMattingService:
    def __init__(
        self,
        *,
        repo_path: Path | str = Path("MODNet"),
        checkpoint_path: Path | str = Path("models/modnet_photographic_portrait_matting.ckpt"),
        prefer_cuda: bool = True,
        backend_factory: Any | None = None,
        logger: Any | None = None,
    ) -> None:
        self.repo_path = Path(repo_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.prefer_cuda = prefer_cuda
        self._backend_factory = backend_factory or (
            lambda device: LocalModnetBackend(
                repo_path=self.repo_path,
                checkpoint_path=self.checkpoint_path,
                device=device,
            )
        )
        self._logger = logger or print
        self.backend = self._initialize_backend()
        self.runtime = ModnetRuntimeInfo(
            device=self.backend.device_name,
            repo_path=str(self.repo_path),
            checkpoint_path=str(self.checkpoint_path),
        )

    def _initialize_backend(self) -> ModnetBackend:
        if self.prefer_cuda and _is_cuda_available():
            try:
                backend = self._backend_factory("cuda")
                self._logger("[MODNet] device=cuda")
                return backend
            except Exception as exc:
                self._logger(f"[MODNet] cuda init failed, fallback to cpu: {exc}")
        backend = self._backend_factory("cpu")
        self._logger("[MODNet] device=cpu")
        return backend

    def matte_image_file(self, source_path: Path, instance_result: InstanceSegmentationResult) -> AlphaConstraintResult:
        try:
            with Image.open(source_path) as image:
                rgb = image.convert("RGB")
        except Exception as exc:
            raise ModnetMattingError(f"Failed to read source image: {source_path} ({exc})", stage="read") from exc
        raw_alpha = self.backend.predict_alpha(rgb)
        if raw_alpha.shape != (rgb.height, rgb.width):
            raw_alpha = cv2.resize(raw_alpha, rgb.size, interpolation=cv2.INTER_LINEAR)
        if int(np.count_nonzero(raw_alpha > 0)) <= 0:
            raise ModnetMattingError("MODNet produced empty alpha.", stage="matting")
        return apply_instance_alpha_constraints(rgb, raw_alpha, instance_result)


def apply_instance_alpha_constraints(
    image: Image.Image,
    raw_alpha: np.ndarray,
    instance_result: InstanceSegmentationResult,
) -> AlphaConstraintResult:
    rgb = image.convert("RGB")
    alpha = raw_alpha.astype(np.uint8)
    if alpha.shape != (rgb.height, rgb.width):
        alpha = cv2.resize(alpha, rgb.size, interpolation=cv2.INTER_LINEAR).astype(np.uint8)

    constrained = alpha.copy()
    sure_fg = instance_result.sure_foreground.astype(bool)
    sure_bg = instance_result.sure_background.astype(bool)
    constrained[sure_fg] = 255
    constrained[sure_bg] = 0

    arr = np.array(rgb.convert("RGBA"))
    arr[:, :, 3] = constrained
    return AlphaConstraintResult(
        image=Image.fromarray(arr, "RGBA"),
        raw_alpha=alpha,
        constrained_alpha=constrained,
        forced_foreground_px=int(np.count_nonzero(sure_fg)),
        forced_background_px=int(np.count_nonzero(sure_bg)),
        unknown_px=int(np.count_nonzero(instance_result.unknown)),
    )
