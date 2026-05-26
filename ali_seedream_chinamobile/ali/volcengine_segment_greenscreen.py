# -*- coding: utf-8 -*-
"""
鐏北寮曟搸鏂硅垷 Seedream锛氬浘鐢熷浘缁垮箷棰勫鐞嗭紙涓?``ali_oss_segment_pipeline_util`` 瀵规帴锛夈€?

渚濊禆::

    pip install 'volcengine-python-sdk[ark]'

榛樿渚濇浣跨敤锛氬叆鍙?``api_key``銆佺幆澧冨彉閲?``ARK_API_KEY``銆?
``volcengine_segment)greenscreen`` 妯″潡鍐?``DEFAULT_ARK_API_KEY``銆?
"""
from __future__ import annotations

import base64
import importlib.util
import os
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np

if TYPE_CHECKING:
    from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = _ROOT / "output"

_DEFAULT_KEY_MODULE_PATH = Path(__file__).resolve().parent / "volcengine_segment)greenscreen.py"


def _load_default_ark_api_key() -> str:
    spec = importlib.util.spec_from_file_location(
        "_volcengine_segment_greenscreen_keys",
        _DEFAULT_KEY_MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        return ""
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    v = getattr(mod, "DEFAULT_ARK_API_KEY", None)
    return str(v).strip() if v else ""


DEFAULT_ARK_API_KEY = '${ARK_API_KEY}'

DEFAULT_GREENSCREEN_PROMPT = (
    "鍥剧墖涓崰鎹浘鐗囨鍓嶆柟鐨勪汉鏄富瑕佺洰鏍囷紝鍙繚鐣欏浘鐗囦腑鐨勪富瑕佺洰鏍囷紝灏嗙洰鏍囨暣浣撳垎鍓插嚭鏉ワ紝"
    "鐩爣杈圭紭涓嶈鏈夎櫄褰辨垨閿娇锛屾渶缁堢粰鎴戜竴寮犲彧鏈変富瑕佺洰鏍囦汉鍍忕殑缁垮箷鍥剧墖锛岀豢骞曞簳鑹茶绾豢锛?锛?55锛?锛?"
)


class VolcengineGreenscreenSegment:
    """PIL 鎴?BGR/BGRA ndarray 鈫?Seedream 缁垮箷鍥?鈫?``cv2`` ndarray銆?""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
        model: str = "doubao-seedream-5-0-260128",
        output_dir: Union[str, Path, None] = None,
    ) -> None:
        key = (api_key or os.environ.get("ARK_API_KEY") or DEFAULT_ARK_API_KEY).strip()
        if not key:
            raise ValueError(
                "璇蜂紶鍏?api_key锛屾垨璁剧疆鐜鍙橀噺 ARK_API_KEY锛屾垨鍦?volcengine_segment)greenscreen.py 涓厤缃?DEFAULT_ARK_API_KEY"
            )
        try:
            from volcenginesdkarkruntime import Ark
        except ImportError as exc:
            raise ImportError(
                "volcengine-python-sdk is required only when use_volcengine_greenscreen is enabled. "
                "Install with: pip install volcengine-python-sdk[ark]"
            ) from exc
        self._client = Ark(base_url=base_url, api_key=key)
        self._model = model
        self._output_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR

    @staticmethod
    def _to_bgr(image: Union["Image.Image", np.ndarray]) -> np.ndarray:
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            if image.ndim == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            if image.shape[2] == 3:
                return np.ascontiguousarray(image)
            raise ValueError(f"涓嶆敮鎸佺殑 ndarray 褰㈢姸: {image.shape}")

        from PIL import Image as PILImage

        if not isinstance(image, PILImage.Image):
            raise TypeError("image 椤讳负 PIL.Image.Image 鎴?numpy.ndarray")

        if image.mode in ("RGBA", "LA"):
            rgba = np.array(image.convert("RGBA"))
            return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        rgb = np.array(image.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _bgr_to_jpeg_data_url(bgr: np.ndarray, *, jpeg_quality: int = 92) -> str:
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if not ok:
            raise RuntimeError("cv2.imencode JPEG 澶辫触")
        b64 = base64.standard_b64encode(buf.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    @staticmethod
    def _download_image_bytes(url: str) -> bytes:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; volcengine-test/1.0)"})
        with urlopen(req, timeout=120) as resp:
            data = resp.read()
        if not data:
            raise RuntimeError("涓嬭浇鐨勫浘鐗囨暟鎹负绌?)
        return data

    @staticmethod
    def _bytes_to_bgr_ndarray(data: bytes) -> np.ndarray:
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError("cv2.imdecode 鏃犳硶瑙ｆ瀽鐢熸垚鍥?)
        return img

    def process(
        self,
        image: Union["Image.Image", np.ndarray],
        *,
        prompt: Optional[str] = None,
        save: bool = False,
        save_path: Union[str, Path, None] = None,
        save_stem: str = "volcengine_greenscreen",
        sequential_image_generation: str = "disabled",
        response_format: str = "url",
        size: str = "2K",
        stream: bool = False,
        watermark: bool = True,
        model: Optional[str] = None,
    ) -> np.ndarray:
        bgr = self._to_bgr(image)
        image_input = self._bgr_to_jpeg_data_url(bgr)
        use_model = model if model is not None else self._model
        text = prompt if prompt is not None else DEFAULT_GREENSCREEN_PROMPT

        images_response = self._client.images.generate(
            model=use_model,
            prompt=text,
            image=image_input,
            sequential_image_generation=sequential_image_generation,
            response_format=response_format,
            size=size,
            stream=stream,
            watermark=watermark,
        )

        if not images_response.data:
            raise RuntimeError("鏂硅垷 API 鏈繑鍥?data")
        url = images_response.data[0].url
        if not url:
            raise RuntimeError("鏂硅垷 API 鏈繑鍥炲浘鐗?URL")

        try:
            raw = self._download_image_bytes(url)
        except (HTTPError, URLError) as e:
            raise RuntimeError(f"涓嬭浇鐢熸垚鍥剧墖澶辫触: {e}") from e

        out_bgr = self._bytes_to_bgr_ndarray(raw)

        if save:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            out = Path(save_path) if save_path is not None else (
                self._output_dir / f"{save_stem}_{uuid.uuid4().hex}.png"
            )
            if not cv2.imwrite(str(out), out_bgr):
                raise RuntimeError(f"鏃犳硶鍐欏叆: {out}")
            print(f"Saved: {out.resolve()}")

        return out_bgr

