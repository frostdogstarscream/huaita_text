from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
from PIL import Image

from ali import AliOssSegmentPipeline as _AliPipeline


class AliSegmentError(RuntimeError):
    def __init__(self, message: str, *, provider: str = "ali_segment_body", stage: str = "segment") -> None:
        super().__init__(message)
        self.provider = provider
        self.stage = stage


def _create_pipeline(api_cfg: dict[str, Any]) -> _AliPipeline:
    """根据 matting_api 布尔开关选择管线：use_suxiaoban > use_seedream > 默认 ali。"""
    use_suxiaoban = bool(api_cfg.get("use_suxiaoban", False))
    use_seedream = bool(api_cfg.get("use_seedream", False))

    output_dir = api_cfg.get("output_dir")
    bucket = api_cfg.get("bucket")
    region = api_cfg.get("region")
    oss_endpoint = api_cfg.get("oss_endpoint")
    imageseg_endpoint = api_cfg.get("imageseg_endpoint")

    base_kwargs: dict[str, Any] = {}
    if output_dir:
        # 解析为绝对路径，避免被管线内部以自身包目录为基准拼接
        base_kwargs["output_dir"] = str(Path(output_dir).resolve())
    if bucket:
        base_kwargs["bucket"] = bucket
    if region:
        base_kwargs["region"] = region
    if oss_endpoint:
        base_kwargs["oss_endpoint"] = oss_endpoint
    if imageseg_endpoint:
        base_kwargs["imageseg_endpoint"] = imageseg_endpoint

    if use_suxiaoban:
        try:
            from ali_seedream_chinamobile.ali import AliOssSegmentPipeline as SuxiaobanPipeline
        except ImportError as exc:
            raise ImportError(
                "ali_seedream_chinamobile 依赖未安装，请执行: pip install pyyaml requests"
            ) from exc

        base_kwargs["ali_segment_max_side"] = int(api_cfg.get("max_image_edge", 2000))
        base_kwargs["use_suxiaoban_greenscreen"] = True
        base_kwargs["use_yolo_segment"] = False
        base_kwargs["use_volcengine_greenscreen"] = False

        sb_cfg = api_cfg.get("suxiaoban", {})
        if isinstance(sb_cfg, dict):
            if sb_cfg.get("api_url"):
                base_kwargs["suxiaoban_api_url"] = str(sb_cfg["api_url"])
            if sb_cfg.get("authorization"):
                base_kwargs["suxiaoban_authorization"] = str(sb_cfg["authorization"])
            if sb_cfg.get("model"):
                base_kwargs["suxiaoban_model"] = str(sb_cfg["model"])
            if sb_cfg.get("prompt"):
                base_kwargs["suxiaoban_prompt"] = str(sb_cfg["prompt"])
            if sb_cfg.get("size"):
                base_kwargs["suxiaoban_size"] = str(sb_cfg["size"])
            if sb_cfg.get("request_timeout_sec"):
                base_kwargs["suxiaoban_request_timeout_sec"] = float(sb_cfg["request_timeout_sec"])

        return _wrap_pipeline_init(SuxiaobanPipeline, base_kwargs, "Suxiaoban")  # type: ignore[arg-type]

    if use_seedream:
        try:
            from ali_seedream.ali import AliOssSegmentPipeline as SeedreamPipeline
        except ImportError as exc:
            raise ImportError(
                "ali_seedream 依赖未安装，请执行: pip install pyyaml volcengine-python-sdk[ark] ultralytics"
            ) from exc

        base_kwargs["ali_segment_max_side"] = int(api_cfg.get("max_image_edge", 2000))
        return _wrap_pipeline_init(SeedreamPipeline, base_kwargs, "Seedream")  # type: ignore[arg-type]

    base_kwargs["max_image_edge"] = int(api_cfg.get("max_image_edge", 2000))
    return _wrap_pipeline_init(_AliPipeline, base_kwargs, "Ali")


def _wrap_pipeline_init(factory: Any, kwargs: dict[str, Any], label: str) -> Any:
    try:
        return factory(**kwargs)
    except AliSegmentError:
        raise
    except Exception as exc:
        raise AliSegmentError(f"Failed to initialize {label} matting pipeline: {exc}", stage="initialize") from exc


class AliSegmentService:
    def __init__(self, config: dict[str, Any]):
        api_cfg = config.get("matting_api", {})
        self.provider = str(api_cfg.get("provider", "ali_segment_body"))

        try:
            self.pipeline = _create_pipeline(api_cfg)
        except AliSegmentError:
            raise
        except ImportError as exc:
            raise AliSegmentError(f"Failed to import matting pipeline dependencies: {exc}", stage="initialize") from exc
        except Exception as exc:
            raise AliSegmentError(f"Failed to initialize matting pipeline: {exc}", stage="initialize") from exc

    def segment_image_file(self, source_path: Path, output_path: Path) -> Image.Image:
        try:
            result = self.pipeline.process_and_save(
                source_path,
                output_filename=output_path.name,
            )
        except AliSegmentError:
            raise
        except Exception as exc:
            raise AliSegmentError(
                f"{self.provider} person segmentation failed: {exc}",
                provider=self.provider,
                stage="segment",
            ) from exc

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if result.ndim == 2:
            image = Image.fromarray(result)
        elif result.ndim == 3 and result.shape[2] == 4:
            rgba = cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA)
            image = Image.fromarray(rgba)
        elif result.ndim == 3 and result.shape[2] == 3:
            rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
        else:
            raise AliSegmentError(
                f"Unsupported segmented image shape: {result.shape}",
                provider=self.provider,
                stage="decode",
            )

        image.save(output_path, format="PNG")
        return image
