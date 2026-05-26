# -*- coding: utf-8 -*-
"""
端到端测试（cv2 ndarray）：本地文件 → numpy.fromfile + cv2.imdecode → OSS 分片上传 → SegmentBody → 保存 output/。

Windows 上 ``cv2.imread`` 对含中文路径支持差，故用 ``np.fromfile`` + ``cv2.imdecode`` 读取。

与 `test_upload_oss_then_segment_greenscreen.py` 一致，仅在「上传来源」处改为传入 OpenCV 解码后的 ndarray。

运行（在项目根目录）：
  python ali/test_upload_oss_then_segment_greenscreen_cv2.py
或：
  python -m ali.test_upload_oss_then_segment_greenscreen_cv2
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

import alibabacloud_oss_v2 as oss
import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ali.ali_oss_multipart_upload_util import AliOssMultipartUploader
from ali.ali_segment_greenscreen_util import AliSegmentBodyGreenscreen

LOCAL_IMAGE = _ROOT / "resource" / "person_front" / "人像照片_低清_缩放符合阿里输入.jpg"
OUTPUT_DIR = _ROOT / "output"
OSS_OBJECT_KEY = "person_front/e2e_test_cv2_ndarray_upload.jpg"
GREEN_OUTPUT_NAME = "e2e_cv2_ndarray_greenscreen.png"

_PRESIGN_GET_EXPIRES = datetime.timedelta(hours=1)


def main() -> None:
    if not LOCAL_IMAGE.is_file():
        raise FileNotFoundError(f"本地图片不存在: {LOCAL_IMAGE}")

    raw = np.fromfile(LOCAL_IMAGE, dtype=np.uint8)
    img_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"cv2.imdecode 无法解码图片: {LOCAL_IMAGE}")

    uploader = AliOssMultipartUploader()
    location_url = uploader.upload_local_file(img_bgr, OSS_OBJECT_KEY)
    print(f"上传完成，对象 location: {location_url}")

    presigned = uploader.client.presign(
        oss.GetObjectRequest(
            bucket=uploader.bucket,
            key=OSS_OBJECT_KEY,
        ),
        expires=_PRESIGN_GET_EXPIRES,
    )
    segment_image_url = presigned.url or ""
    if not segment_image_url:
        raise RuntimeError("生成预签名 GET URL 失败")
    print(f"SegmentBody 拉取原图 URL（预签名）: {segment_image_url}")

    segmenter = AliSegmentBodyGreenscreen()
    out_path = OUTPUT_DIR / GREEN_OUTPUT_NAME
    bgr = segmenter.segment_oss_url_to_bgr(segment_image_url)
    saved = segmenter.save_bgr(bgr, out_path)
    resolved = saved.resolve()
    print(f"合成绿幕图片保存位置: {resolved}")


if __name__ == "__main__":
    main()
