# -*- coding: utf-8 -*-
"""
速销班网关图生图（对齐 Ark ``/images/generations`` 请求体结构）。

依赖: pip install requests opencv-python pillow

``SuxiaobanImageGenerationsClient``：输入 ``PIL.Image`` / ``numpy.ndarray``（cv2 BGR/BGRA/灰度）/
本地图片路径，调用网关后返回 ``numpy.ndarray``（解码后的 BGR 或 BGRA），并默认保存到 ``output/``。

通过 ``AliOssSegmentPipeline`` 运行时，网关 URL、鉴权、模型、提示词等以
``ali/ali_oss_segment_pipeline_config.yaml`` 中 ``suxiaoban_*`` 为准，并由
``ali_oss_segment_pipeline_util`` 读入后传入本类构造函数；下方 ``DEFAULT_*`` 仅在
独立调用本类或 YAML 缺省时作为兜底。

注意: 勿将含密钥的脚本或 YAML 提交到公开仓库；生产环境请改用环境变量或私密配置源。
"""
from __future__ import annotations

import base64
import json
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import cv2
import numpy as np
import requests

if TYPE_CHECKING:
    from PIL.Image import Image as PILImageType

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DEFAULT_OUTPUT_DIR = _ROOT / "output"

# --- 默认网关配置（实例化时可覆盖）---
DEFAULT_API_URL = (
    "https://ai.suxiaoban.com:18000/llmgate/SE012026000061/v3/images/generations"
)
DEFAULT_AUTHORIZATION = (
    "cVumS#ITKv3jypgcJ2g6E6jqLK&Ud%llTUc#@R$!&Q!0HFysO5szs#Y9DvD7"
)
DEFAULT_PROMPT = (
    "图片中占据图片正前方的人是主要目标，只保留图片中的主要目标，将目标整体分割出来，目标边缘不要有虚影或锯齿，最终给我一张只有主要目标人像的绿幕图片，绿幕底色要纯绿（0，255，0）"
)
DEFAULT_MODEL = "doubao-seedream-4-5-251128"

DEFAULT_RESPONSE_FORMAT = "url"
DEFAULT_SIZE = "2K"
DEFAULT_WATERMARK = True
DEFAULT_SEQUENTIAL_IMAGE_GENERATION = "disabled"
DEFAULT_STREAM = False
DEFAULT_REQUEST_TIMEOUT_SEC = 180

# 便于 ``python ali/test_suxiaoban_image_generations.py`` 快速试跑
LOCAL_IMAGE_PATH = _ROOT / "resource" / "person_front" / "重叠人像_1500x2000.jpg"


def _build_body(
    *,
    model: str,
    prompt: str,
    image_data_url: str,
    response_format: Optional[str],
    size: Optional[str],
    seed: Optional[int],
    guidance_scale: Optional[float],
    watermark: Optional[bool],
    optimize_prompt: Optional[bool],
    optimize_prompt_options: Optional[Dict[str, Any]],
    sequential_image_generation: Optional[str],
    sequential_image_generation_options: Optional[Dict[str, Any]],
    tools: Optional[Any],
    output_format: Optional[str],
    stream: Any,
) -> Dict[str, Any]:
    raw: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "image": image_data_url,
        "response_format": response_format,
        "size": size,
        "seed": seed,
        "guidance_scale": guidance_scale,
        "watermark": watermark,
        "optimize_prompt": optimize_prompt,
        "optimize_prompt_options": optimize_prompt_options,
        "sequential_image_generation": sequential_image_generation,
        "sequential_image_generation_options": sequential_image_generation_options,
        "tools": tools,
        "output_format": output_format,
        "stream": stream,
    }
    return {k: v for k, v in raw.items() if v is not None}


class SuxiaobanImageGenerationsClient:
    """速销班 llmgate 图生图：多类型输入 → 网关 → ``uint8`` ndarray（BGR 或 BGRA）+ 磁盘保存。"""

    def __init__(
        self,
        *,
        api_url: str = DEFAULT_API_URL,
        authorization: str = DEFAULT_AUTHORIZATION,
        output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
        model: str = DEFAULT_MODEL,
        prompt: str = DEFAULT_PROMPT,
        response_format: Optional[str] = DEFAULT_RESPONSE_FORMAT,
        size: Optional[str] = DEFAULT_SIZE,
        seed: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        watermark: Optional[bool] = DEFAULT_WATERMARK,
        optimize_prompt: Optional[bool] = None,
        sequential_image_generation: Optional[str] = DEFAULT_SEQUENTIAL_IMAGE_GENERATION,
        output_format: Optional[str] = None,
        stream: Any = DEFAULT_STREAM,
        request_timeout_sec: float = DEFAULT_REQUEST_TIMEOUT_SEC,
    ) -> None:
        self._api_url = api_url
        self._authorization = authorization
        self._output_dir = Path(output_dir)
        self._model = model
        self._default_prompt = prompt
        self._response_format = response_format
        self._size = size
        self._seed = seed
        self._guidance_scale = guidance_scale
        self._watermark = watermark
        self._optimize_prompt = optimize_prompt
        self._sequential_image_generation = sequential_image_generation
        self._output_format = output_format
        self._stream = stream
        self._request_timeout_sec = request_timeout_sec

    @staticmethod
    def _pil_to_bgr(image: "PILImageType") -> np.ndarray:
        from PIL import Image as PILImage

        if not isinstance(image, PILImage.Image):
            raise TypeError("须为 PIL.Image.Image")
        if image.mode in ("RGBA", "LA"):
            rgba = np.array(image.convert("RGBA"))
            return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        rgb = np.array(image.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _load_as_bgr(
        self,
        image: Union[str, Path, np.ndarray, "PILImageType"],
    ) -> np.ndarray:
        """路径 / ndarray / PIL → BGR ``uint8``（四通道输入会压成三通道再送 JPEG）。"""
        if isinstance(image, np.ndarray):
            arr = np.ascontiguousarray(image)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            if arr.ndim == 2:
                return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            if arr.ndim == 3 and arr.shape[2] == 4:
                return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            if arr.ndim == 3 and arr.shape[2] == 3:
                return arr
            raise ValueError(f"不支持的 ndarray 形状: {arr.shape}")

        from PIL import Image as PILImage

        if isinstance(image, PILImage.Image):
            return self._pil_to_bgr(image)

        path = Path(image)
        if not path.is_file():
            raise FileNotFoundError(f"本地文件不存在: {path.resolve()}")
        raw = np.fromfile(str(path), dtype=np.uint8)
        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if bgr is not None:
            return bgr
        with PILImage.open(path) as im:
            return self._pil_to_bgr(im.copy())

    @staticmethod
    def _bgr_to_jpeg_data_url(bgr: np.ndarray, *, jpeg_quality: int = 92) -> str:
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if not ok:
            raise RuntimeError("cv2.imencode JPEG 失败")
        b64 = base64.standard_b64encode(buf.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    def _request_once(
        self,
        image_data_url: str,
        use_prompt: str,
        verbose: bool,
    ) -> np.ndarray:
        """单次请求：POST → 解析 → 下载 → 返回 ndarray。"""
        body = _build_body(
            model=self._model,
            prompt=use_prompt,
            image_data_url=image_data_url,
            response_format=self._response_format,
            size=self._size,
            seed=self._seed,
            guidance_scale=self._guidance_scale,
            watermark=self._watermark,
            optimize_prompt=self._optimize_prompt,
            optimize_prompt_options=None,
            sequential_image_generation=self._sequential_image_generation,
            sequential_image_generation_options=None,
            tools=None,
            output_format=self._output_format,
            stream=self._stream,
        )

        headers = {
            "Authorization": self._authorization,
            "Content-Type": "application/json",
        }

        if verbose:
            print(f"POST {self._api_url}")
            print(f"model={self._model}, body_keys={list(body.keys())}")

        resp = requests.post(
            self._api_url,
            headers=headers,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            timeout=self._request_timeout_sec,
        )

        if verbose:
            print(f"HTTP {resp.status_code}")

        try:
            payload = resp.json()
        except json.JSONDecodeError:
            print(f"[Suxiaoban] 响应非 JSON: {resp.text[:2000]}")
            resp.raise_for_status()
            raise

        if resp.status_code != 200:
            print(f"[Suxiaoban] HTTP {resp.status_code} 响应: {json.dumps(payload, ensure_ascii=False, indent=2)[:2000]}")
            resp.raise_for_status()

        data_list = payload.get("data") or []
        if not data_list:
            raise RuntimeError(f"响应中无 data: {payload}")

        url = data_list[0].get("url")
        if not url:
            raise RuntimeError(f"响应中无 url: {data_list[0]}")

        if verbose:
            print("下载生成图:", url[:120] + ("..." if len(url) > 120 else ""))

        img_resp = requests.get(url, timeout=self._request_timeout_sec)
        img_resp.raise_for_status()

        raw_bytes = img_resp.content
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        out = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if out is None:
            raise RuntimeError("无法解码网关返回的图片数据")
        return out

    def generate(
        self,
        image: Union[str, Path, np.ndarray, "PILImageType"],
        *,
        prompt: Optional[str] = None,
        save: bool = True,
        save_path: Optional[Union[str, Path]] = None,
        save_stem: str = "suxiaoban_seedream",
        jpeg_quality: int = 92,
        verbose: bool = True,
        retries: int = 3,
    ) -> np.ndarray:
        """
        Args:
            image: 本地路径、BGR/BGRA/灰度 ``ndarray``，或 ``PIL.Image``。
            prompt: 覆盖构造时的默认提示词。
            save: 是否写入 ``output_dir``。
            retries: 5xx 错误时重试次数。

        Returns:
            网关返回图像解码后的 ``numpy.ndarray``。
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)

        bgr_in = self._load_as_bgr(image)
        image_data_url = self._bgr_to_jpeg_data_url(bgr_in, jpeg_quality=jpeg_quality)
        use_prompt = prompt if prompt is not None else self._default_prompt

        last_exc: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                out = self._request_once(image_data_url, use_prompt, verbose)
            except requests.HTTPError as exc:
                last_exc = exc
                if exc.response is not None and exc.response.status_code >= 500 and attempt < retries:
                    wait = 2.0 * attempt
                    print(f"[Suxiaoban] 5xx 错误，{wait:.0f}s 后重试 ({attempt}/{retries}): {exc}")
                    import time as _time
                    _time.sleep(wait)
                    continue
                raise
            except Exception as exc:
                last_exc = exc
                if attempt < retries:
                    wait = 2.0 * attempt
                    print(f"[Suxiaoban] 请求异常，{wait:.0f}s 后重试 ({attempt}/{retries}): {exc}")
                    import time as _time
                    _time.sleep(wait)
                    continue
                raise

            if save:
                if save_path is not None:
                    out_file = Path(save_path)
                else:
                    ext = ".jpg"
                    out_file = self._output_dir / f"{save_stem}_{uuid.uuid4().hex}{ext}"
                out_file.parent.mkdir(parents=True, exist_ok=True)
                ok = cv2.imwrite(str(out_file), out)
                if not ok:
                    out_file.write_bytes(cv2.imencode(".jpg", out)[1].tobytes())
                if verbose:
                    print(f"已保存: {out_file.resolve()}")
            return out

        raise last_exc  # type: ignore[misc]


def main() -> None:
    path = Path(LOCAL_IMAGE_PATH)
    if not path.is_file():
        raise FileNotFoundError(f"示例图片不存在: {path.resolve()}")
    client = SuxiaobanImageGenerationsClient()
    _ = client.generate(path)


if __name__ == "__main__":
    main()
