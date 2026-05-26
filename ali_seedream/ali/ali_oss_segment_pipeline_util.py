# -*- coding: utf-8 -*-
"""
OSS 上传 + SegmentBody 人像分割流水线工具。

默认从 ``ali/ali_oss_segment_pipeline_config.yaml`` 读取构造参数（需 ``pip install pyyaml``），
也可以在 ``AliOssSegmentPipeline(...)`` 中传入非 ``None`` 的实参覆盖对应项。

封装 ``AliOssMultipartUploader`` 与 ``AliSegmentBodyGreenscreen``：支持本地路径、
OpenCV ndarray、PIL Image 输入；可选 YOLO 最大实例抠图、可选火山方舟 Seedream 绿幕预处理后，
再上传到 OSS → 预签名 GET → 阿里云人像分割 → 保存项目 ``output/``，
（送入分割前最长边由配置 ``ali_segment_max_side`` 控制，含 OSS 前对火山 2K 等结果的再缩放）。
返回 ``numpy.ndarray``（cv2 约定的 BGR 或带 Alpha 的 BGRA）。
"""
from __future__ import annotations

import datetime
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import alibabacloud_oss_v2 as oss
import cv2
import numpy as np

from ali.ali_oss_multipart_upload_util import (
    DEFAULT_ACCESS_KEY_ID,
    DEFAULT_ACCESS_KEY_SECRET,
    DEFAULT_BUCKET,
    DEFAULT_ENDPOINT,
    DEFAULT_REGION,
    AliOssMultipartUploader,
)
from ali.ali_segment_greenscreen_util import AliSegmentBodyGreenscreen
from .volcengine_segment_greenscreen import DEFAULT_ARK_API_KEY
# VolcengineGreenscreenSegment / YoloLargestSegmentMask 延迟到 __init__ 中按需导入

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT_DIR = _ROOT / "output"
# 未在 YAML 中配置 ``ali_segment_max_side`` 时的默认值（见配置文件说明）
_DEFAULT_ALI_SEGMENT_MAX_SIDE = 1999
# 兼容旧代码对模块常量的引用；实际以 YAML / ``self._ali_segment_max_side`` 为准
ALI_SEGMENT_MAX_SIDE = _DEFAULT_ALI_SEGMENT_MAX_SIDE


def _default_pipeline_config_path() -> Path:
    return Path(__file__).resolve().parent / "ali_oss_segment_pipeline_config.yaml"


def _load_pipeline_yaml_mapping(
    config_path: Union[str, Path, None] = None,
) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "读取流水线配置需要 PyYAML，请执行: pip install pyyaml"
        ) from e

    path = Path(config_path) if config_path is not None else _default_pipeline_config_path()
    if not path.is_file():
        raise FileNotFoundError(f"流水线 YAML 配置不存在: {path.resolve()}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML 根节点须为 mapping，当前为: {type(raw).__name__}")
    return raw


def build_pipeline_init_kwargs_from_yaml(
    config_path: Union[str, Path, None] = None,
) -> Dict[str, Any]:
    """
    从 YAML 构建与 ``AliOssSegmentPipeline.__init__`` 兼容的关键字参数（不含 ``config_path``）。
    缺省字段回退到 ``ali_oss_multipart_upload_util`` / ``volcengine_segment`` 等模块内的默认值。
    """
    m = _load_pipeline_yaml_mapping(config_path)

    output_dir_val = m.get("output_dir")
    if output_dir_val is None or output_dir_val == "":
        output_dir: Optional[Path] = _DEFAULT_OUTPUT_DIR
    else:
        p = Path(str(output_dir_val))
        output_dir = p if p.is_absolute() else (_ROOT / p)

    seg_ep = m.get("imageseg_endpoint")
    imageseg_endpoint: Optional[str] = None if seg_ep in (None, "") else str(seg_ep)

    sec = m.get("presign_expires_seconds", 3600)
    presign_expires = datetime.timedelta(seconds=int(sec))

    return {
        "access_key_id": str(m.get("access_key_id") or DEFAULT_ACCESS_KEY_ID),
        "access_key_secret": str(m.get("access_key_secret") or DEFAULT_ACCESS_KEY_SECRET),
        "bucket": str(m.get("bucket") or DEFAULT_BUCKET),
        "region": str(m.get("region") or DEFAULT_REGION),
        "oss_endpoint": str(m.get("oss_endpoint") or DEFAULT_ENDPOINT),
        "imageseg_endpoint": imageseg_endpoint,
        "output_dir": output_dir,
        "presign_expires": presign_expires,
        "use_yolo_segment": bool(m.get("use_yolo_segment", False)),
        "use_volcengine_greenscreen": bool(m.get("use_volcengine_greenscreen", True)),
        "volcengine_api_key": str(m.get("volcengine_api_key") or DEFAULT_ARK_API_KEY),
        "ali_segment_max_side": int(m.get("ali_segment_max_side", _DEFAULT_ALI_SEGMENT_MAX_SIDE)),
    }


if TYPE_CHECKING:
    from PIL.Image import Image as PILImageType


class AliOssSegmentPipeline:
    """统一入口：路径 / ndarray / PIL →（可选 YOLO、可选火山绿幕）→ OSS → SegmentBody → 写入 ``output`` → 返回 ndarray。"""

    def __init__(
        self,
        *,
        config_path: Union[str, Path, None] = None,
        access_key_id: Optional[str] = None,
        access_key_secret: Optional[str] = None,
        bucket: Optional[str] = None,
        region: Optional[str] = None,
        oss_endpoint: Optional[str] = None,
        imageseg_endpoint: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        presign_expires: Optional[datetime.timedelta] = None,
        use_yolo_segment: Optional[bool] = None,
        use_volcengine_greenscreen: Optional[bool] = None,
        volcengine_api_key: Optional[str] = None,
        ali_segment_max_side: Optional[int] = None,
    ) -> None:
        yk = build_pipeline_init_kwargs_from_yaml(config_path)
        access_key_id = yk["access_key_id"] if access_key_id is None else access_key_id
        access_key_secret = yk["access_key_secret"] if access_key_secret is None else access_key_secret
        bucket = yk["bucket"] if bucket is None else bucket
        region = yk["region"] if region is None else region
        oss_endpoint = yk["oss_endpoint"] if oss_endpoint is None else oss_endpoint
        if imageseg_endpoint is None:
            imageseg_endpoint = yk["imageseg_endpoint"]
        if output_dir is None:
            out_dir = yk["output_dir"]
        else:
            p = Path(output_dir)
            out_dir = p if p.is_absolute() else (_ROOT / p)
        presign_expires = yk["presign_expires"] if presign_expires is None else presign_expires
        use_yolo_segment = yk["use_yolo_segment"] if use_yolo_segment is None else use_yolo_segment
        use_volcengine_greenscreen = (
            yk["use_volcengine_greenscreen"]
            if use_volcengine_greenscreen is None
            else use_volcengine_greenscreen
        )
        volcengine_api_key = yk["volcengine_api_key"] if volcengine_api_key is None else volcengine_api_key
        self._ali_segment_max_side = (
            yk["ali_segment_max_side"] if ali_segment_max_side is None else int(ali_segment_max_side)
        )

        self._uploader = AliOssMultipartUploader(
            access_key_id=access_key_id,
            access_key_secret=access_key_secret,
            bucket=bucket,
            region=region,
            endpoint=oss_endpoint,
        )
        self._segmenter = AliSegmentBodyGreenscreen(
            access_key_id=access_key_id,
            access_key_secret=access_key_secret,
            endpoint=imageseg_endpoint,
        )
        self._output_dir = Path(out_dir)
        self._presign_expires = presign_expires
        self._use_yolo_segment = use_yolo_segment
        self._yolo_mask = None
        if use_yolo_segment:
            from .yolo_seg import YoloLargestSegmentMask
            self._yolo_mask = YoloLargestSegmentMask(output_dir=self._output_dir)

        self._use_volcengine_greenscreen = use_volcengine_greenscreen
        self._volcengine = None
        if use_volcengine_greenscreen:
            from .volcengine_segment_greenscreen import VolcengineGreenscreenSegment
            self._volcengine = VolcengineGreenscreenSegment(
                api_key=volcengine_api_key,
                output_dir=self._output_dir,
            )

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

    def _load_image_for_yolo(
        self,
        image: Union[str, Path, np.ndarray, "PILImageType"],
    ) -> Union[np.ndarray, "PILImageType"]:
        if isinstance(image, np.ndarray):
            return image
        from PIL import Image as PILImage

        if isinstance(image, PILImage.Image):
            return image
        path = Path(image)
        if not path.is_file():
            raise FileNotFoundError(f"本地文件不存在: {path.resolve()}")
        bgr = cv2.imread(str(path))
        if bgr is not None:
            return bgr
        with PILImage.open(path) as im:
            return im.copy()

    @staticmethod
    def _scale_down_if_exceeds_max_side(
        img: np.ndarray,
        max_side: int = ALI_SEGMENT_MAX_SIDE,
    ) -> np.ndarray:
        """等比缩放，使 ``max(宽, 高) <= max_side``（仅当原图超长边大于 max_side 时缩放）。"""
        if img.ndim == 2:
            h, w = img.shape
        elif img.ndim == 3:
            h, w = img.shape[:2]
        else:
            raise ValueError(f"不支持的 ndarray 形状: {img.shape}")

        if max(h, w) <= max_side:
            return img

        scale = max_side / max(h, w)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    def _prepare_input_ndarray(
        self,
        image: Union[str, Path, np.ndarray, "PILImageType"],
        *,
        max_side: Optional[int] = None,
    ) -> np.ndarray:
        """路径 / PIL / ndarray → ``uint8`` ndarray（灰度 / BGR / BGRA），并满足分割接口边长上限。"""
        ms = self._ali_segment_max_side if max_side is None else max_side
        if isinstance(image, np.ndarray):
            arr = np.ascontiguousarray(image)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return self._scale_down_if_exceeds_max_side(arr, ms)

        from PIL import Image as PILImage

        if isinstance(image, PILImage.Image):
            if image.mode in ("RGBA", "LA"):
                rgba = np.array(image.convert("RGBA"))
                arr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
            else:
                rgb = np.array(image.convert("RGB"))
                arr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return self._scale_down_if_exceeds_max_side(arr, ms)

        path = Path(image)
        if not path.is_file():
            raise FileNotFoundError(f"本地文件不存在: {path.resolve()}")
        raw = np.fromfile(str(path), dtype=np.uint8)
        arr = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
        if arr is None:
            with PILImage.open(path) as im:
                return self._prepare_input_ndarray(im.copy(), max_side=ms)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return self._scale_down_if_exceeds_max_side(arr, ms)

    def _to_array_or_pil_for_volcengine(
        self,
        upload_image: Union[str, Path, np.ndarray, "PILImageType"],
    ) -> Union[np.ndarray, "PILImageType"]:
        if isinstance(upload_image, np.ndarray):
            return upload_image
        from PIL import Image as PILImage

        if isinstance(upload_image, PILImage.Image):
            return upload_image
        return self._load_image_for_yolo(upload_image)

    def _coerce_upload_for_aliyun_imageseg(
        self,
        upload_image: Union[np.ndarray, "PILImageType"],
        *,
        max_side: Optional[int] = None,
    ) -> np.ndarray:
        """YOLO/火山等步骤之后、OSS 上传之前：统一为 ``uint8`` ndarray 并限制最长边（含 2K 出图过大）。"""
        ms = self._ali_segment_max_side if max_side is None else max_side
        from PIL import Image as PILImage

        if isinstance(upload_image, PILImage.Image):
            if upload_image.mode in ("RGBA", "LA"):
                rgba = np.array(upload_image.convert("RGBA"))
                arr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
            else:
                rgb = np.array(upload_image.convert("RGB"))
                arr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(upload_image, np.ndarray):
            arr = np.ascontiguousarray(upload_image)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            raise TypeError(
                f"上传前仅支持 ndarray 或 PIL.Image，收到: {type(upload_image).__name__}"
            )
        return self._scale_down_if_exceeds_max_side(arr, ms)

    def process_and_save(
        self,
        image: Union[str, Path, np.ndarray, "PILImageType"],
        *,
        oss_object_key: Optional[str] = None,
        output_filename: Optional[str] = None,
        upload_verbose: bool = False,
    ) -> np.ndarray:
        """
        上传 ``image`` 至 OSS：可选 YOLO 最大实例 BGRA 抠图，可选火山 Seedream 绿幕预处理
        （顺序：YOLO → 火山 → OSS）。再使用预签名 URL 调阿里云人像分割，将结果写入 ``output_dir``。

        送入 OSS / 阿里云分割前会限制分辨率：最长边不超过构造时的 ``ali_segment_max_side``
        （默认来自 ``ali_oss_segment_pipeline_config.yaml``）；输入会先缩放，**火山等中间步骤若产出更大图，
        在上传 OSS 前会再次缩放**。

        Args:
            image: 本地路径、BGR/BGRA/灰度 ``numpy.ndarray``（cv2）、或 ``PIL.Image.Image``。
            oss_object_key: OSS 对象 Key；未指定时，若开启 YOLO 或火山绿幕则默认 ``...png``，否则 ``...jpg``。
            output_filename: 相对 ``output_dir`` 的文件名；默认 ``pipeline_segment_<uuid>.png``。
            upload_verbose: 是否打印 OSS 分片上传明细日志。

        Returns:
            分割后的图像 ``numpy.ndarray``（BGR 或 BGRA，dtype uint8）。
        """
        use_png_key = self._use_yolo_segment or self._use_volcengine_greenscreen
        if oss_object_key is not None:
            object_key = oss_object_key
        elif use_png_key:
            object_key = f"person_front/pipeline_{uuid.uuid4().hex}.png"
        else:
            object_key = f"person_front/pipeline_{uuid.uuid4().hex}.jpg"
        filename = output_filename or f"pipeline_segment_{uuid.uuid4().hex}.png"

        self._ensure_output_dir()

        work = self._prepare_input_ndarray(image)

        if self._use_yolo_segment:
            assert self._yolo_mask is not None
            upload_image = self._yolo_mask.process(work, save=False)
        else:
            upload_image = work

        if self._use_volcengine_greenscreen:
            assert self._volcengine is not None
            v_in = self._to_array_or_pil_for_volcengine(upload_image)
            upload_image = self._volcengine.process(v_in, save=False)

        upload_image = self._coerce_upload_for_aliyun_imageseg(upload_image)

        self._uploader.upload_local_file(
            upload_image,
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
