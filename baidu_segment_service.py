import base64
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image


class BaiduSegmentError(RuntimeError):
    pass


DEMO_BAIDU_API_KEY = "eelQQQCNaabdlZ8hlAVWOkdk"
DEMO_BAIDU_SECRET_KEY = "GIquqj496DQefeWwHmWXXOmi5kgsPCWr"


class BaiduSegmentService:
    def __init__(self, config: dict[str, Any]):
        api_cfg = config.get("matting_api", {})
        self.token_url = api_cfg.get("token_url", "https://aip.baidubce.com/oauth/2.0/token")
        self.segment_url = api_cfg.get(
            "segment_url",
            "https://aip.baidubce.com/rest/2.0/image-classify/v1/body_seg",
        )
        self.result_type = api_cfg.get("type", "foreground")
        self.timeout = int(api_cfg.get("timeout_seconds", 60))
        self.retry_times = max(int(api_cfg.get("retry_times", 2)), 0)
        self.use_hardcoded_key = bool(api_cfg.get("use_hardcoded_key", True))
        self._access_token: str | None = None
        self._token_expire_at = 0.0

    def segment_image_file(self, source_path: Path, output_path: Path) -> Image.Image:
        access_token = self.get_access_token()
        image_base64 = self._image_file_to_base64(source_path)
        result_base64 = self._request_segment(image_base64, access_token)
        image = self.decode_rgba_base64_to_image(result_base64)
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format="PNG")
        return image

    def get_access_token(self) -> str:
        now = time.time()
        if self._access_token and now < self._token_expire_at:
            return self._access_token

        api_key = os.getenv("BAIDU_API_KEY")
        secret_key = os.getenv("BAIDU_SECRET_KEY")
        if self.use_hardcoded_key:
            api_key = api_key or DEMO_BAIDU_API_KEY
            secret_key = secret_key or DEMO_BAIDU_SECRET_KEY
        if not api_key or not secret_key:
            raise BaiduSegmentError(
                "Missing Baidu credentials. Set BAIDU_API_KEY / BAIDU_SECRET_KEY or fill DEMO_BAIDU_API_KEY / DEMO_BAIDU_SECRET_KEY in baidu_segment_service.py."
            )

        response = requests.get(
            self.token_url,
            params={
                "grant_type": "client_credentials",
                "client_id": api_key,
                "client_secret": secret_key,
            },
            timeout=self.timeout,
        )
        try:
            payload = response.json()
        except ValueError as exc:
            raise BaiduSegmentError(f"Failed to parse Baidu token response: {response.text[:300]}") from exc
        if response.status_code >= 400:
            raise BaiduSegmentError(f"Failed to get Baidu access token: {payload}")
        token = payload.get("access_token")
        if not token:
            raise BaiduSegmentError(f"Failed to get Baidu access token: {payload}")

        expires_in = int(payload.get("expires_in", 0))
        self._access_token = token
        self._token_expire_at = time.time() + max(expires_in - 60, 60)
        return token

    def _request_segment(self, image_base64: str, access_token: str) -> str:
        params = {"access_token": access_token}
        payload = {
            "image": image_base64,
            "type": self.result_type,
        }
        headers = {"content-type": "application/x-www-form-urlencoded"}

        last_error: Exception | None = None
        for _ in range(self.retry_times + 1):
            try:
                response = requests.post(
                    self.segment_url,
                    params=params,
                    data=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                result = response.json()
                if response.status_code >= 400:
                    raise BaiduSegmentError(f"Baidu body_seg failed: {result}")
                if result.get("error_code"):
                    raise BaiduSegmentError(f"Baidu body_seg failed: {result}")
                if int(result.get("person_num", 0)) <= 0:
                    raise BaiduSegmentError(f"No valid person detected: {result}")
                image_data = result.get("foreground")
                if not image_data:
                    raise BaiduSegmentError(f"Baidu body_seg did not return foreground: {result}")
                return image_data
            except (requests.RequestException, ValueError, BaiduSegmentError) as exc:
                last_error = exc

        if last_error is None:
            raise BaiduSegmentError("Baidu body_seg request failed for unknown reason.")
        if isinstance(last_error, BaiduSegmentError):
            raise last_error
        raise BaiduSegmentError(f"Baidu body_seg request failed: {last_error}") from last_error

    @staticmethod
    def _image_file_to_base64(source_path: Path) -> str:
        return base64.b64encode(source_path.read_bytes()).decode("utf-8")

    @staticmethod
    def decode_rgba_base64_to_image(image_base64: str) -> Image.Image:
        cleaned = image_base64.strip()
        if cleaned.startswith("data:image"):
            cleaned = cleaned.split(",", 1)[1]
        binary = base64.b64decode(cleaned)
        image = Image.open(BytesIO(binary))
        image.load()
        return image
