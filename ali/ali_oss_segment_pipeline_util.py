# -*- coding: utf-8 -*-
"""
OSS 上传 + SegmentBody 人像分割流水线工具。

封装 ``AliOssMultipartUploader`` 与 ``AliSegmentBodyGreenscreen``：支持本地路径、
OpenCV ndarray、PIL Image 输入；上传到 OSS → 预签名 GET → 分割 → 保存项目 ``output/``，
返回 ``numpy.ndarray``（cv2 约定的 BGR 或带 Alpha 的 BGRA）。
"""
from __future__ import annotations

import datetime
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import alibabacloud_oss_v2 as oss
import numpy as np

from ali.ali_oss_multipart_upload_util import (
    DEFAULT_ACCESS_KEY_ID,
    DEFAULT_ACCESS_KEY_SECRET,
    DEFAULT_BUCKET,
    DEFAULT_ENDPOINT,
    DEFAULT_MAX_IMAGE_EDGE,
    DEFAULT_REGION,
    AliOssMultipartUploader,
)
from ali.ali_segment_greenscreen_util import AliSegmentBodyGreenscreen

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT_DIR = _ROOT / "output"

if TYPE_CHECKING:
    from PIL.Image import Image as PILImageType


class AliOssSegmentPipeline:
    """统一入口：路径 / ndarray / PIL → OSS → SegmentBody → 写入 ``output`` → 返回 ndarray。"""

    def __init__(
        self,
        *,
        access_key_id: str = DEFAULT_ACCESS_KEY_ID,
        access_key_secret: str = DEFAULT_ACCESS_KEY_SECRET,
        bucket: str = DEFAULT_BUCKET,
        region: str = DEFAULT_REGION,
        oss_endpoint: str = DEFAULT_ENDPOINT,
        imageseg_endpoint: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        presign_expires: datetime.timedelta = datetime.timedelta(hours=1),
        max_image_edge: int = DEFAULT_MAX_IMAGE_EDGE,
    ) -> None:
        self._uploader = AliOssMultipartUploader(
            access_key_id=access_key_id,
            access_key_secret=access_key_secret,
            bucket=bucket,
            region=region,
            endpoint=oss_endpoint,
            max_image_edge=max_image_edge,
        )
        self._segmenter = AliSegmentBodyGreenscreen(
            access_key_id=access_key_id,
            access_key_secret=access_key_secret,
            endpoint=imageseg_endpoint,
        )
        self._output_dir = Path(output_dir) if output_dir is not None else _DEFAULT_OUTPUT_DIR
        self._presign_expires = presign_expires

    def _ensure_output_dir(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def _presigned_object_url(self, object_key: str) -> str:
        presigned = self._uploader.client.presign(
            oss.GetObjectRequest(
                bucket=self._uploader.bucket,
                key=object_key,
            ),
            expires=self._presign_expires,
        )
        url = presigned.url or ""
        if not url:
            raise RuntimeError("生成预签名 GET URL 失败")
        return url

    def process_and_save(
        self,
        image: Union[str, Path, np.ndarray, "PILImageType"],
        *,
        oss_object_key: Optional[str] = None,
        output_filename: Optional[str] = None,
        upload_verbose: bool = False,
    ) -> np.ndarray:
        """
        上传 ``image`` 至 OSS，使用预签名 URL 调人像分割，将结果写入 ``output_dir``。

        Args:
            image: 本地路径、BGR/BGRA/灰度 ``numpy.ndarray``（cv2）、或 ``PIL.Image.Image``。
            oss_object_key: OSS 对象 Key；默认 ``person_front/pipeline_<uuid>.jpg``。
            output_filename: 相对 ``output_dir`` 的文件名；默认 ``pipeline_segment_<uuid>.png``。
            upload_verbose: 是否打印 OSS 分片上传明细日志。

        Returns:
            分割后的图像 ``numpy.ndarray``（BGR 或 BGRA，dtype uint8）。
        """
        object_key = oss_object_key or f"person_front/pipeline_{uuid.uuid4().hex}.jpg"
        filename = output_filename or f"pipeline_segment_{uuid.uuid4().hex}.png"

        self._ensure_output_dir()

        self._uploader.upload_local_file(
            image,
            object_key,
            verbose=upload_verbose,
        )

        segment_url = self._presigned_object_url(object_key)
        result = self._segmenter.segment_oss_url_to_bgr(segment_url)

        out_path = self._output_dir / filename
        self._segmenter.save_bgr(result, out_path)

        resolved = out_path.resolve()
        h, w = result.shape[:2]
        channels = result.shape[2] if result.ndim == 3 else 1
        print(f"保存完整路径: {resolved}")
        print(f"图片大小: 宽={w}, 高={h}, 通道数={channels}; ndarray.shape={result.shape}, dtype={result.dtype}")

        return result
