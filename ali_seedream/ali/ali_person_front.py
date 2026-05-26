# -*- coding: utf-8 -*-
# This file is auto-generated, don't edit it. Thanks.
import os
import sys
import json
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from typing import List

import cv2
import numpy as np
import requests
from alibabacloud_imageseg20191230.client import Client as imageseg20191230Client
from alibabacloud_credentials.client import Client as CredentialClient
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_imageseg20191230 import models as imageseg_20191230_models
from alibabacloud_tea_util import models as util_models
from alibabacloud_tea_util.client import Client as UtilClient
from alibabacloud_credentials.models import Config as CredentialConfig


class Sample:
    def __init__(self):
        pass

    @staticmethod
    def create_client() -> imageseg20191230Client:
        """
        浣跨敤鍑嵁鍒濆鍖栬处鍙稢lient
        @return: Client
        @throws Exception
        """
        credentialsConfig = CredentialConfig(
            type='access_key',
            # 蹇呭～鍙傛暟锛屾澶勪互浠庣幆澧冨彉閲忎腑鑾峰彇AccessKey ID涓轰緥
            access_key_id='${ALIYUN_ACCESS_KEY_ID}',
            # 蹇呭～鍙傛暟锛屾澶勪互浠庣幆澧冨彉閲忎腑鑾峰彇AccessKey Secret涓轰緥
            access_key_secret='${ALIYUN_ACCESS_KEY_SECRET}'
        )
        credentialsClient = CredentialClient(credentialsConfig)
        config = open_api_models.Config(
            credential=credentialsClient,
            endpoint='<endpoint>'
        )
        
        # Endpoint 璇峰弬鑰?https://api.aliyun.com/product/imageseg
        config.endpoint = f'imageseg.cn-shanghai.aliyuncs.com'
        return imageseg20191230Client(config)

    @staticmethod
    def _resp_to_dict(resp) -> dict:
        if hasattr(resp, "to_map"):
            return resp.to_map()
        if isinstance(resp, dict):
            return resp
        return json.loads(json.dumps(resp, default=str))

    @staticmethod
    def _download_image_to_output(image_url: str, prefix: str = "ali_segment") -> str:
        output_dir = Path(__file__).resolve().parent.parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        parsed = urlparse(image_url)
        ext = Path(parsed.path).suffix or ".png"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = output_dir / f"{prefix}_{ts}{ext}"

        r = requests.get(image_url, timeout=30)
        r.raise_for_status()
        img_arr = np.frombuffer(r.content, dtype=np.uint8)
        # 浜哄儚鍒嗗壊甯镐负甯﹂€忔槑閫氶亾 PNG锛汭MREAD_COLOR 浼氫涪鎺?alpha锛?
        # 閫忔槑澶勪細鍙樻垚榛戣壊锛屸€滆繎鐧介槇鍊尖€濇浛鎹篃鍖归厤涓嶅埌銆?
        img = cv2.imdecode(img_arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            out_path.write_bytes(r.content)
            return str(out_path)

        green_bgr = np.array([[[0.0, 255.0, 0.0]]], dtype=np.float32)

        if img.ndim == 3 and img.shape[2] == 4:
            bgr = img[:, :, :3].astype(np.float32)
            a = img[:, :, 3:4].astype(np.float32) / 255.0
            img = np.clip(bgr * a + green_bgr * (1.0 - a), 0, 255).astype(np.uint8)
        else:
            # 鏃?Alpha 鏃朵粛鎸夋祬鑹茶儗鏅厹搴曚负缁垮箷
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.ndim == 3:
                img = img[:, :, :3]
            white_mask = np.all(img >= 245, axis=2)
            img = img.copy()
            img[white_mask] = (0, 255, 0)

        ok, buf = cv2.imencode(ext, img)
        if not ok:
            raise RuntimeError("淇濆瓨缁垮箷鍥惧け璐ワ細cv2.imencode 澶辫触")
        out_path.write_bytes(buf.tobytes())
        return str(out_path)

    @staticmethod
    def main(
        args: List[str],
    ) -> None:
        client = Sample.create_client()
        segment_body_request = imageseg_20191230_models.SegmentBodyRequest(
            image_url='https://example.com/person_front_sample.jpg'
        )
        runtime = util_models.RuntimeOptions()
        try:
            resp = client.segment_body_with_options(segment_body_request, runtime)
            print(json.dumps(resp, default=str, indent=2))
            resp_map = Sample._resp_to_dict(resp)
            image_url = (((resp_map.get("body") or {}).get("Data") or {}).get("ImageURL"))
            if image_url:
                saved_path = Sample._download_image_to_output(image_url, prefix="ali_segment_sync")
                print(f"ImageURL 涓嬭浇瀹屾垚: {saved_path}")
            else:
                print("鍝嶅簲涓湭鎵惧埌 body.Data.ImageURL")
        except Exception as error:
            # 姝ゅ浠呭仛鎵撳嵃灞曠ず锛岃璋ㄦ厧瀵瑰緟寮傚父澶勭悊锛屽湪宸ョ▼椤圭洰涓垏鍕跨洿鎺ュ拷鐣ュ紓甯搞€?
            # 閿欒 message
            print(error.message)
            # 璇婃柇鍦板潃
            print(error.data.get("Recommend"))

    @staticmethod
    async def main_async(
        args: List[str],
    ) -> None:
        client = Sample.create_client()
        segment_body_request = imageseg_20191230_models.SegmentBodyRequest(
            image_url='https://example.com/person_front_sample.jpg'
        )
        runtime = util_models.RuntimeOptions()
        try:
            resp = await client.segment_body_with_options_async(segment_body_request, runtime)
            print(json.dumps(resp, default=str, indent=2))
            resp_map = Sample._resp_to_dict(resp)
            image_url = (((resp_map.get("body") or {}).get("Data") or {}).get("ImageURL"))
            if image_url:
                saved_path = Sample._download_image_to_output(image_url, prefix="ali_segment_async")
                print(f"ImageURL 涓嬭浇瀹屾垚: {saved_path}")
            else:
                print("鍝嶅簲涓湭鎵惧埌 body.Data.ImageURL")
        except Exception as error:
            # 姝ゅ浠呭仛鎵撳嵃灞曠ず锛岃璋ㄦ厧瀵瑰緟寮傚父澶勭悊锛屽湪宸ョ▼椤圭洰涓垏鍕跨洿鎺ュ拷鐣ュ紓甯搞€?
            # 閿欒 message
            print(error.message)
            # 璇婃柇鍦板潃
            print(error.data.get("Recommend"))


if __name__ == '__main__':
    Sample.main(sys.argv[1:])
