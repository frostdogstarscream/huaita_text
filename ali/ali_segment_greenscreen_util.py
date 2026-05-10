# -*- coding: utf-8 -*-
"""
阿里云图像分割人体接口（SegmentBody）+ 绿幕合成工具。

参照 `ali_person_front.py` 的调用方式，将 OSS/公网图片 URL 传入接口，
解析返回 `body.Data.ImageURL`，下载分割结果：含 Alpha 时保留原图，否则将浅色底替换为绿幕 BGR。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
import requests
from alibabacloud_credentials.client import Client as CredentialClient
from alibabacloud_credentials.models import Config as CredentialConfig
from alibabacloud_imageseg20191230.client import Client as ImageSegClient
from alibabacloud_imageseg20191230 import models as imageseg_20191230_models
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_tea_util import models as util_models

# 与 `ali_oss_test.py` 保持一致；调用方可传入其它密钥覆盖
DEFAULT_ACCESS_KEY_ID = os.environ.get("ALI_ACCESS_KEY_ID", "")
DEFAULT_ACCESS_KEY_SECRET = os.environ.get("ALI_ACCESS_KEY_SECRET", "")


class AliSegmentBodyGreenscreen:
    """将对象存储等平台上的图片 URL 传入阿里云人像分割接口；RGBA 保留原图，无 Alpha 时产出绿幕 BGR。"""

    DEFAULT_ENDPOINT = "imageseg.cn-shanghai.aliyuncs.com"

    def __init__(
        self,
        access_key_id: str = DEFAULT_ACCESS_KEY_ID,
        access_key_secret: str = DEFAULT_ACCESS_KEY_SECRET,
        endpoint: Optional[str] = None,
    ):
        if not access_key_id or not access_key_secret:
            raise ValueError("缺少 AccessKey：请传入有效的 access_key_id 与 access_key_secret")

        cred_cfg = CredentialConfig(
            type="access_key",
            access_key_id=access_key_id,
            access_key_secret=access_key_secret,
        )

        credentials_client = CredentialClient(cred_cfg)
        api_cfg = open_api_models.Config(
            credential=credentials_client,
            endpoint=endpoint or self.DEFAULT_ENDPOINT,
        )
        self._client = ImageSegClient(api_cfg)
        self._runtime = util_models.RuntimeOptions()

    @staticmethod
    def response_to_map(resp: Any) -> Dict[str, Any]:
        if hasattr(resp, "to_map"):
            return resp.to_map()
        if isinstance(resp, dict):
            return resp
        return json.loads(json.dumps(resp, default=str))

    @staticmethod
    def extract_result_image_url(resp_map: Dict[str, Any]) -> str:
        body = resp_map.get("body") or {}
        data = body.get("Data") or {}
        url = data.get("ImageURL")
        if not url:
            raise ValueError("响应中未找到 body.Data.ImageURL")
        return str(url)

    @staticmethod
    def composite_green_from_bytes(content: bytes) -> np.ndarray:
        """
        解码接口返回的分割图。若为带 Alpha 的四通道图（BGRA），直接返回解码结果；
        否则将浅色背景（近似白）替换为纯绿 BGR(0,255,0)。
        """
        arr = np.frombuffer(content, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("无法解码分割结果图片")

        if img.ndim == 3 and img.shape[2] == 4:
            return img

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3:
            img = img[:, :, :3].copy()
        else:
            raise ValueError(f"不支持的图像维度: {img.shape}")

        white_mask = np.all(img >= 245, axis=2)
        img[white_mask] = (0, 255, 0)
        return img

    def download_result_green_bgr(self, result_image_url: str, timeout: float = 30.0) -> np.ndarray:
        r = requests.get(result_image_url, timeout=timeout)
        r.raise_for_status()
        return self.composite_green_from_bytes(r.content)

    def save_bgr(
        self,
        bgr: np.ndarray,
        out_path: Union[str, Path],
        ext_hint: Optional[str] = None,
    ) -> Path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        ext = ext_hint or p.suffix.lower() or ".png"
        if ext and not ext.startswith("."):
            ext = "." + ext
        ok, buf = cv2.imencode(ext, bgr)
        if not ok:
            raise RuntimeError(f"cv2.imencode 失败: {ext}")
        p.write_bytes(buf.tobytes())
        return p

    def segment_oss_url_to_bgr(self, image_url: str) -> np.ndarray:
        """同步：公网/OSS 可访问的图片 URL → 分割结果（RGBA 为四通道原图，否则为绿幕 BGR）。"""
        req = imageseg_20191230_models.SegmentBodyRequest(image_url=image_url)
        resp = self._client.segment_body_with_options(req, self._runtime)
        mp = self.response_to_map(resp)
        result_url = self.extract_result_image_url(mp)
        arr = self.download_result_green_bgr(result_url)
        print(f"分割结果 ndarray 尺寸: {arr.shape}")
        return arr

    async def segment_oss_url_to_bgr_async(self, image_url: str) -> np.ndarray:
        """异步版本；返回值含义同 segment_oss_url_to_bgr。"""
        req = imageseg_20191230_models.SegmentBodyRequest(image_url=image_url)
        resp = await self._client.segment_body_with_options_async(req, self._runtime)
        mp = self.response_to_map(resp)
        result_url = self.extract_result_image_url(mp)
        arr = self.download_result_green_bgr(result_url)
        print(f"分割结果 ndarray 尺寸: {arr.shape}")
        return arr

    def segment_oss_url_to_file(
        self,
        image_url: str,
        out_path: Union[str, Path],
    ) -> Path:
        """同步：分割并保存；RGBA 写 PNG 保留 Alpha，否则按路径后缀编码。"""
        bgr = self.segment_oss_url_to_bgr(image_url)
        return self.save_bgr(bgr, out_path)
