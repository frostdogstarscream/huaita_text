# -*- coding: utf-8 -*-
"""
``AliOssSegmentPipeline`` 调用示例（对齐三套端到端测试脚本的输入方式）。

1. **路径**：与 ``test_upload_oss_then_segment_greenscreen.py`` 相同，直接传 ``Path``。
2. **cv2 ndarray**：与 ``test_upload_oss_then_segment_greenscreen_cv2.py`` 相同，
   Windows 中文路径用 ``np.fromfile`` + ``cv2.imdecode``，再传入流水线。
3. **PIL**：与 ``test_upload_oss_then_segment_greenscreen_pil.py`` 相同，``Image.open`` 后传入流水线。

运行（在项目根目录）::

    python ali/example_oss_segment_pipeline.py
    python ali/example_oss_segment_pipeline.py --mode path
    python ali/example_oss_segment_pipeline.py --mode cv2
    python ali/example_oss_segment_pipeline.py --mode pil

默认 ``--mode all`` 会依次执行三种示例（各调用一次云端接口，耗时与计费较多）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ali.ali_oss_segment_pipeline_util import AliOssSegmentPipeline

LOCAL_IMAGE = _ROOT / "resource" / "person_front" / "人像照片_低清_缩放符合阿里输入.jpg"

# 与各测试脚本中的 OSS Key / 输出文件名对齐，便于对照产物
OSS_KEY_PATH = "person_front/人像照片_低清_缩放符合阿里输入.jpg"
OUT_NAME_PATH = "人像照片_低清_缩放符合阿里输入_greenscreen.png"

OSS_KEY_CV2 = "person_front/e2e_test_cv2_ndarray_upload.jpg"
OUT_NAME_CV2 = "e2e_cv2_ndarray_greenscreen.png"

OSS_KEY_PIL = "person_front/e2e_test_pil_image_upload.jpg"
OUT_NAME_PIL = "e2e_pil_image_greenscreen.png"


def _ensure_sample_exists() -> None:
    if not LOCAL_IMAGE.is_file():
        raise FileNotFoundError(f"示例图片不存在: {LOCAL_IMAGE}")


def example_path(pipeline: AliOssSegmentPipeline, *, upload_verbose: bool = False) -> None:
    print("\n=== 示例 1：本地路径 Path/str（同 test_upload_oss_then_segment_greenscreen.py）===")
    arr = pipeline.process_and_save(
        LOCAL_IMAGE,
        oss_object_key=OSS_KEY_PATH,
        output_filename=OUT_NAME_PATH,
        upload_verbose=upload_verbose,
    )
    print(f"[示例1] 返回 ndarray shape={arr.shape}")


def example_cv2(pipeline: AliOssSegmentPipeline, *, upload_verbose: bool = False) -> None:
    print("\n=== 示例 2：cv2 ndarray（同 test_upload_oss_then_segment_greenscreen_cv2.py）===")
    raw = np.fromfile(LOCAL_IMAGE, dtype=np.uint8)
    img_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"cv2.imdecode 失败: {LOCAL_IMAGE}")

    arr = pipeline.process_and_save(
        img_bgr,
        oss_object_key=OSS_KEY_CV2,
        output_filename=OUT_NAME_CV2,
        upload_verbose=upload_verbose,
    )
    print(f"[示例2] 返回 ndarray shape={arr.shape}")


def example_pil(pipeline: AliOssSegmentPipeline, *, upload_verbose: bool = False) -> None:
    print("\n=== 示例 3：PIL.Image（同 test_upload_oss_then_segment_greenscreen_pil.py）===")
    pil_img = Image.open(LOCAL_IMAGE)
    arr = pipeline.process_and_save(
        pil_img,
        oss_object_key=OSS_KEY_PIL,
        output_filename=OUT_NAME_PIL,
        upload_verbose=upload_verbose,
    )
    print(f"[示例3] 返回 ndarray shape={arr.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AliOssSegmentPipeline 调用示例")
    parser.add_argument(
        "--mode",
        choices=("path", "cv2", "pil", "all"),
        default="all",
        help="path / cv2 / pil 跑单一路径；all 依次跑三种（默认）",
    )
    parser.add_argument(
        "--upload-verbose",
        action="store_true",
        help="打印 OSS 分片上传明细日志",
    )
    args = parser.parse_args()

    _ensure_sample_exists()

    pipeline = AliOssSegmentPipeline(output_dir=_ROOT / "output")

    mode = args.mode
    uv = args.upload_verbose
    if mode in ("path", "all"):
        example_path(pipeline, upload_verbose=uv)
    if mode in ("cv2", "all"):
        example_cv2(pipeline, upload_verbose=uv)
    if mode in ("pil", "all"):
        example_pil(pipeline, upload_verbose=uv)


if __name__ == "__main__":
    main()
