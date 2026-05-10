from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
from PIL import Image

from ali import AliOssSegmentPipeline


class AliSegmentError(RuntimeError):
    pass


class AliSegmentService:
    def __init__(self, config: dict[str, Any]):
        api_cfg = config.get("matting_api", {})
        self.provider = str(api_cfg.get("provider", "ali_segment_body"))
        self.output_dir = api_cfg.get("output_dir")
        self.bucket = api_cfg.get("bucket")
        self.region = api_cfg.get("region")
        self.oss_endpoint = api_cfg.get("oss_endpoint")
        self.imageseg_endpoint = api_cfg.get("imageseg_endpoint")
        self.max_image_edge = int(api_cfg.get("max_image_edge", 2000))

        pipeline_kwargs: dict[str, Any] = {
            "max_image_edge": self.max_image_edge,
        }
        if self.output_dir:
            pipeline_kwargs["output_dir"] = self.output_dir
        if self.bucket:
            pipeline_kwargs["bucket"] = self.bucket
        if self.region:
            pipeline_kwargs["region"] = self.region
        if self.oss_endpoint:
            pipeline_kwargs["oss_endpoint"] = self.oss_endpoint
        if self.imageseg_endpoint:
            pipeline_kwargs["imageseg_endpoint"] = self.imageseg_endpoint

        self.pipeline = AliOssSegmentPipeline(**pipeline_kwargs)

    def segment_image_file(self, source_path: Path, output_path: Path) -> Image.Image:
        try:
            result = self.pipeline.process_and_save(
                source_path,
                output_filename=output_path.name,
            )
        except Exception as exc:
            raise AliSegmentError(f"Aliyun person segmentation failed: {exc}") from exc

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
            raise AliSegmentError(f"Unsupported segmented image shape: {result.shape}")

        image.save(output_path, format="PNG")
        return image
