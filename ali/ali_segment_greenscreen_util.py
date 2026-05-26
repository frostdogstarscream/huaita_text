# -*- coding: utf-8 -*-
"""
闃块噷浜戝浘鍍忓垎鍓蹭汉浣撴帴鍙ｏ紙SegmentBody锛? 缁垮箷鍚堟垚宸ュ叿銆?

鍙傜収 `ali_person_front.py` 鐨勮皟鐢ㄦ柟寮忥紝灏?OSS/鍏綉鍥剧墖 URL 浼犲叆鎺ュ彛锛?
瑙ｆ瀽杩斿洖 `body.Data.ImageURL`锛屼笅杞藉垎鍓茬粨鏋滐細鍚?Alpha 鏃朵繚鐣欏師鍥撅紝鍚﹀垯灏嗘祬鑹插簳鏇挎崲涓虹豢骞?BGR銆?
"""
from __future__ import annotations

import json
import time
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

# 涓?`ali_oss_test.py` 淇濇寔涓€鑷达紱璋冪敤鏂瑰彲浼犲叆鍏跺畠瀵嗛挜瑕嗙洊
DEFAULT_ACCESS_KEY_ID = "${ALIYUN_ACCESS_KEY_ID}"
DEFAULT_ACCESS_KEY_SECRET = "${ALIYUN_ACCESS_KEY_SECRET}"


class AliSegmentBodyGreenscreen:
    """灏嗗璞″瓨鍌ㄧ瓑骞冲彴涓婄殑鍥剧墖 URL 浼犲叆闃块噷浜戜汉鍍忓垎鍓叉帴鍙ｏ紱RGBA 淇濈暀鍘熷浘锛屾棤 Alpha 鏃朵骇鍑虹豢骞?BGR銆?""

    DEFAULT_ENDPOINT = "imageseg.cn-shanghai.aliyuncs.com"

    def __init__(
        self,
        access_key_id: str = DEFAULT_ACCESS_KEY_ID,
        access_key_secret: str = DEFAULT_ACCESS_KEY_SECRET,
        endpoint: Optional[str] = None,
    ):
        if not access_key_id or not access_key_secret:
            raise ValueError("缂哄皯 AccessKey锛氳浼犲叆鏈夋晥鐨?access_key_id 涓?access_key_secret")

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
            raise ValueError("鍝嶅簲涓湭鎵惧埌 body.Data.ImageURL")
        return str(url)

    @staticmethod
    def composite_green_from_bytes(content: bytes) -> np.ndarray:
        """
        瑙ｇ爜鎺ュ彛杩斿洖鐨勫垎鍓插浘銆傝嫢涓哄甫 Alpha 鐨勫洓閫氶亾鍥撅紙BGRA锛夛紝鐩存帴杩斿洖瑙ｇ爜缁撴灉锛?
        鍚﹀垯灏嗘祬鑹茶儗鏅紙杩戜技鐧斤級鏇挎崲涓虹函缁?BGR(0,255,0)銆?
        """
        arr = np.frombuffer(content, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("鏃犳硶瑙ｇ爜鍒嗗壊缁撴灉鍥剧墖")

        if img.ndim == 3 and img.shape[2] == 4:
            return img

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3:
            img = img[:, :, :3].copy()
        else:
            raise ValueError(f"涓嶆敮鎸佺殑鍥惧儚缁村害: {img.shape}")

        white_mask = np.all(img >= 245, axis=2)
        img[white_mask] = (0, 255, 0)
        return img

    def download_result_green_bgr(self, result_image_url: str, timeout: float = 30.0) -> np.ndarray:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                r = requests.get(result_image_url, timeout=timeout)
                r.raise_for_status()
                return self.composite_green_from_bytes(r.content)
            except requests.HTTPError as exc:
                if exc.response is not None and 500 <= exc.response.status_code < 600 and attempt < 2:
                    time.sleep(1.0 * (attempt + 1))
                    last_error = exc
                    continue
                raise
            except requests.RequestException:
                if attempt < 2:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                raise
        raise last_error  # type: ignore[misc]

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
            raise RuntimeError(f"cv2.imencode 澶辫触: {ext}")
        p.write_bytes(buf.tobytes())
        return p

    def segment_oss_url_to_bgr(self, image_url: str) -> np.ndarray:
        """鍚屾锛氬叕缃?OSS 鍙闂殑鍥剧墖 URL 鈫?鍒嗗壊缁撴灉锛圧GBA 涓哄洓閫氶亾鍘熷浘锛屽惁鍒欎负缁垮箷 BGR锛夈€?""
        req = imageseg_20191230_models.SegmentBodyRequest(image_url=image_url)
        resp = self._client.segment_body_with_options(req, self._runtime)
        mp = self.response_to_map(resp)
        result_url = self.extract_result_image_url(mp)
        arr = self.download_result_green_bgr(result_url)
        print(f"鍒嗗壊缁撴灉 ndarray 灏哄: {arr.shape}")
        return arr

    async def segment_oss_url_to_bgr_async(self, image_url: str) -> np.ndarray:
        """寮傛鐗堟湰锛涜繑鍥炲€煎惈涔夊悓 segment_oss_url_to_bgr銆?""
        req = imageseg_20191230_models.SegmentBodyRequest(image_url=image_url)
        resp = await self._client.segment_body_with_options_async(req, self._runtime)
        mp = self.response_to_map(resp)
        result_url = self.extract_result_image_url(mp)
        arr = self.download_result_green_bgr(result_url)
        print(f"鍒嗗壊缁撴灉 ndarray 灏哄: {arr.shape}")
        return arr

    def segment_oss_url_to_file(
        self,
        image_url: str,
        out_path: Union[str, Path],
    ) -> Path:
        """鍚屾锛氬垎鍓插苟淇濆瓨锛汻GBA 鍐?PNG 淇濈暀 Alpha锛屽惁鍒欐寜璺緞鍚庣紑缂栫爜銆?""
        bgr = self.segment_oss_url_to_bgr(image_url)
        return self.save_bgr(bgr, out_path)

