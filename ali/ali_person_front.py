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
        使用凭据初始化账号Client
        @return: Client
        @throws Exception
        """
        credentialsConfig = CredentialConfig(
            type='access_key',
            # 必填参数，此处以从环境变量中获取AccessKey ID为例
            access_key_id=os.environ.get('ALI_ACCESS_KEY_ID', ''),
            # 必填参数，此处以从环境变量中获取AccessKey Secret为例
            access_key_secret=os.environ.get('ALI_ACCESS_KEY_SECRET', '')
        )
        credentialsClient = CredentialClient(credentialsConfig)
        config = open_api_models.Config(
            credential=credentialsClient,
            endpoint='<endpoint>'
        )
        
        # Endpoint 请参考 https://api.aliyun.com/product/imageseg
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
        # 人像分割常为带透明通道 PNG；IMREAD_COLOR 会丢掉 alpha，
        # 透明处会变成黑色，“近白阈值”替换也匹配不到。
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
            # 无 Alpha 时仍按浅色背景兜底为绿幕
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.ndim == 3:
                img = img[:, :, :3]
            white_mask = np.all(img >= 245, axis=2)
            img = img.copy()
            img[white_mask] = (0, 255, 0)

        ok, buf = cv2.imencode(ext, img)
        if not ok:
            raise RuntimeError("保存绿幕图失败：cv2.imencode 失败")
        out_path.write_bytes(buf.tobytes())
        return str(out_path)

    @staticmethod
    def main(
        args: List[str],
    ) -> None:
        client = Sample.create_client()
        segment_body_request = imageseg_20191230_models.SegmentBodyRequest(
            image_url='https://huaita-person-img.oss-cn-shanghai.aliyuncs.com/person_front/%E4%BA%BA%E5%83%8F%E7%85%A7%E7%89%87_%E4%BD%8E%E6%B8%85_%E7%BC%A9%E6%94%BE%E7%AC%A6%E5%90%88%E9%98%BF%E9%87%8C%E8%BE%93%E5%85%A5.jpg?Expires=1778129483&OSSAccessKeyId=TMP.3Ku6mV984rgvTUtNNp5RzNUa1FEtAMpWsJnqjV5BYHSRWjTiXY8rsu5uZnTE2a5Cb3He7HhXk5yGvSn4sNVLBJJSs7hnqB&Signature=M7ouVDKCEJ%2FQqRKVXFoao3TmkoQ%3D'
        )
        runtime = util_models.RuntimeOptions()
        try:
            resp = client.segment_body_with_options(segment_body_request, runtime)
            print(json.dumps(resp, default=str, indent=2))
            resp_map = Sample._resp_to_dict(resp)
            image_url = (((resp_map.get("body") or {}).get("Data") or {}).get("ImageURL"))
            if image_url:
                saved_path = Sample._download_image_to_output(image_url, prefix="ali_segment_sync")
                print(f"ImageURL 下载完成: {saved_path}")
            else:
                print("响应中未找到 body.Data.ImageURL")
        except Exception as error:
            # 此处仅做打印展示，请谨慎对待异常处理，在工程项目中切勿直接忽略异常。
            # 错误 message
            print(error.message)
            # 诊断地址
            print(error.data.get("Recommend"))

    @staticmethod
    async def main_async(
        args: List[str],
    ) -> None:
        client = Sample.create_client()
        segment_body_request = imageseg_20191230_models.SegmentBodyRequest(
            image_url='https://huaita-person-img.oss-cn-shanghai.aliyuncs.com/person_front/%E4%BA%BA%E5%83%8F%E7%85%A7%E7%89%87_%E4%BD%8E%E6%B8%85_%E7%BC%A9%E6%94%BE%E7%AC%A6%E5%90%88%E9%98%BF%E9%87%8C%E8%BE%93%E5%85%A5.jpg?Expires=1778122711&OSSAccessKeyId=TMP.3Ku6mV984rgvTUtNNp5RzNUa1FEtAMpWsJnqjV5BYHSRWjTiXY8rsu5uZnTE2a5Cb3He7HhXk5yGvSn4sNVLBJJSs7hnqB&Signature=kQNPdjXSxYg6c7YWLRQyuppaiL0%3D'
        )
        runtime = util_models.RuntimeOptions()
        try:
            resp = await client.segment_body_with_options_async(segment_body_request, runtime)
            print(json.dumps(resp, default=str, indent=2))
            resp_map = Sample._resp_to_dict(resp)
            image_url = (((resp_map.get("body") or {}).get("Data") or {}).get("ImageURL"))
            if image_url:
                saved_path = Sample._download_image_to_output(image_url, prefix="ali_segment_async")
                print(f"ImageURL 下载完成: {saved_path}")
            else:
                print("响应中未找到 body.Data.ImageURL")
        except Exception as error:
            # 此处仅做打印展示，请谨慎对待异常处理，在工程项目中切勿直接忽略异常。
            # 错误 message
            print(error.message)
            # 诊断地址
            print(error.data.get("Recommend"))


if __name__ == '__main__':
    Sample.main(sys.argv[1:])