# -*- coding: utf-8 -*-
"""
端到端测试：本地人像图 → OSS 分片上传 → 人像分割 + 绿幕合成 → 保存到 output/。

上传完成后 Console 中的「位置」即 `upload_local_file` 的返回值（与 CompleteMultipart 的 location 一致）。
若 Bucket 为「私有读」，匿名 location 无法被 SegmentBody 服务端拉取，需对同一对象再生成
预签名 GET URL 传入 `segment_oss_url_to_bgr`（本脚本已自动处理）。

运行（在项目根目录）：
  python ali/test_upload_oss_then_segment_greenscreen.py
或：
  python -m ali.test_upload_oss_then_segment_greenscreen
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

import alibabacloud_oss_v2 as oss

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ali.ali_oss_multipart_upload_util import AliOssMultipartUploader
from ali.ali_segment_greenscreen_util import AliSegmentBodyGreenscreen

# 与仓库资源路径一致
LOCAL_IMAGE = _ROOT / "resource" / "person_front" / "人像照片_低清_缩放符合阿里输入.jpg"
OUTPUT_DIR = _ROOT / "output"
OSS_OBJECT_KEY = "person_front/人像照片_低清_缩放符合阿里输入.jpg"
GREEN_OUTPUT_NAME = "人像照片_低清_缩放符合阿里输入_greenscreen.png"

# 预签名 URL 有效期（SegmentBody 拉取原图用）
_PRESIGN_GET_EXPIRES = datetime.timedelta(hours=1)


def main() -> None:
    if not LOCAL_IMAGE.is_file():
        raise FileNotFoundError(f"本地图片不存在: {LOCAL_IMAGE}")

    uploader = AliOssMultipartUploader()
    location_url = uploader.upload_local_file(LOCAL_IMAGE, OSS_OBJECT_KEY)
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
