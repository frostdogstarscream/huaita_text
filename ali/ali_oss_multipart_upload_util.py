# -*- coding: utf-8 -*-
"""
闃块噷浜?OSS 鍒嗙墖涓婁紶宸ュ叿锛氭敮鎸佹湰鍦拌矾寰勩€丱penCV ndarray銆丳IL Image 杈撳叆锛?
骞跺湪涓婁紶鍓嶆寜鏈€闀胯竟 2000 鍍忕礌杩涜绛夋瘮缂╂斁銆?
"""
from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Union, overload

import alibabacloud_oss_v2 as oss
import cv2
import numpy as np

if TYPE_CHECKING:
    from PIL.Image import Image as PILImageType

try:
    from PIL import Image as PILImageModule
except ImportError:
    PILImageModule = None  # type: ignore[misc, assignment]

DEFAULT_ACCESS_KEY_ID = "${ALIYUN_ACCESS_KEY_ID}"
DEFAULT_ACCESS_KEY_SECRET = "${ALIYUN_ACCESS_KEY_SECRET}"
DEFAULT_BUCKET = "huaita-person-img"
DEFAULT_REGION = "cn-shanghai"
DEFAULT_ENDPOINT = "https://oss-cn-shanghai.aliyuncs.com"
DEFAULT_PART_SIZE = 100 * 1024
DEFAULT_MAX_IMAGE_EDGE = 2000


class AliOssMultipartUploader:
    """灏嗘湰鍦板浘鐗囨垨鍐呭瓨鍥剧墖鍒嗙墖涓婁紶鍒伴樋閲屼簯 OSS銆?""

    def __init__(
        self,
        access_key_id: str = DEFAULT_ACCESS_KEY_ID,
        access_key_secret: str = DEFAULT_ACCESS_KEY_SECRET,
        bucket: str = DEFAULT_BUCKET,
        region: str = DEFAULT_REGION,
        endpoint: str = DEFAULT_ENDPOINT,
        part_size: int = DEFAULT_PART_SIZE,
        max_image_edge: int = DEFAULT_MAX_IMAGE_EDGE,
    ) -> None:
        if not access_key_id or not access_key_secret:
            raise ValueError("缂哄皯 access_key_id / access_key_secret")

        credentials_provider = oss.credentials.StaticCredentialsProvider(
            access_key_id,
            access_key_secret,
        )
        config = oss.config.load_default()
        config.credentials_provider = credentials_provider
        config.region = region
        config.endpoint = endpoint

        self._client = oss.Client(config)
        self._bucket = bucket
        self._part_size = part_size
        self._max_image_edge = max_image_edge

    @property
    def client(self) -> oss.Client:
        return self._client

    @property
    def bucket(self) -> str:
        return self._bucket

    def _resize_cv2_ndarray_if_needed(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim < 2:
            raise ValueError(f"涓嶆敮鎸佺殑 ndarray 褰㈢姸: {arr.shape}")
        height, width = arr.shape[:2]
        max_edge = max(width, height)
        if max_edge <= self._max_image_edge:
            return arr
        scale = self._max_image_edge / float(max_edge)
        new_width = max(int(round(width * scale)), 1)
        new_height = max(int(round(height * scale)), 1)
        return cv2.resize(arr, (new_width, new_height), interpolation=cv2.INTER_AREA)

    def _resize_pil_image_if_needed(self, im: PILImageType) -> PILImageType:
        width, height = im.size
        max_edge = max(width, height)
        if max_edge <= self._max_image_edge:
            return im
        scale = self._max_image_edge / float(max_edge)
        new_width = max(int(round(width * scale)), 1)
        new_height = max(int(round(height * scale)), 1)
        resampling = getattr(PILImageModule, "Resampling", None)
        lanczos = resampling.LANCZOS if resampling is not None else PILImageModule.LANCZOS
        return im.resize((new_width, new_height), lanczos)

    @staticmethod
    def _decode_local_image_path(path: Path) -> np.ndarray:
        raw = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"鏃犳硶瑙ｇ爜鏈湴鍥剧墖: {path}")
        return img

    def _encode_cv2_ndarray_to_bytes(self, arr: np.ndarray) -> bytes:
        if not isinstance(arr, np.ndarray):
            raise TypeError("搴斾负 numpy.ndarray")
        img = arr
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        img = self._resize_cv2_ndarray_if_needed(img)
        if img.ndim == 2:
            ext = ".png"
        elif img.ndim == 3 and img.shape[2] == 4:
            ext = ".png"
        elif img.ndim == 3 and img.shape[2] == 3:
            ext = ".jpg"
        else:
            raise ValueError(f"涓嶆敮鎸佺殑 ndarray 褰㈢姸: {img.shape}")
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            raise RuntimeError(f"cv2.imencode 澶辫触: {ext}")
        return buf.tobytes()

    def _encode_pil_image_to_bytes(self, im: PILImageType) -> bytes:
        im = self._resize_pil_image_if_needed(im)
        bio = BytesIO()
        fmt = "PNG" if im.mode in ("RGBA", "LA", "P") else "JPEG"
        im.save(bio, format=fmt)
        data = bio.getvalue()
        if not data:
            raise RuntimeError("PIL 缂栫爜缁撴灉涓虹┖")
        return data

    def _multipart_upload_with_reader(
        self,
        object_key: str,
        file_size: int,
        read_part: Callable[[int, int], bytes],
        *,
        verbose: bool,
    ) -> str:
        bucket = self._bucket

        initiate_result = self._client.initiate_multipart_upload(
            oss.InitiateMultipartUploadRequest(
                bucket=bucket,
                key=object_key,
            )
        )

        upload_id = initiate_result.upload_id
        if verbose:
            print(
                f"鍒濆鍖栧垎鐗囦笂浼犳垚鍔燂紝鐘舵€佺爜:{initiate_result.status_code}, "
                f"璇锋眰ID:{initiate_result.request_id}, 涓婁紶ID:{upload_id}"
            )

        part_size = self._part_size
        part_number = 1
        upload_parts = []
        offset = 0

        while offset < file_size:
            current_part_size = min(part_size, file_size - offset)
            part_data = read_part(offset, current_part_size)

            part_result = self._client.upload_part(
                oss.UploadPartRequest(
                    bucket=bucket,
                    key=object_key,
                    upload_id=upload_id,
                    part_number=part_number,
                    body=part_data,
                )
            )

            if verbose:
                print(
                    f"鐘舵€佺爜: {part_result.status_code}, 璇锋眰ID: {part_result.request_id}, "
                    f"鍒嗙墖鍙? {part_number}, ETag: {part_result.etag}"
                )

            upload_parts.append(
                oss.UploadPart(
                    part_number=part_number,
                    etag=part_result.etag,
                )
            )

            offset += current_part_size
            part_number += 1

        upload_parts.sort(key=lambda p: p.part_number)

        complete_result = self._client.complete_multipart_upload(
            oss.CompleteMultipartUploadRequest(
                bucket=bucket,
                key=object_key,
                upload_id=upload_id,
                complete_multipart_upload=oss.CompleteMultipartUpload(parts=upload_parts),
            )
        )

        if verbose:
            print(
                f"瀹屾垚鍒嗙墖涓婁紶锛岀姸鎬佺爜:{complete_result.status_code}, "
                f"璇锋眰ID:{complete_result.request_id}, "
                f"Bucket:{complete_result.bucket}, "
                f"Key:{complete_result.key}, "
                f"浣嶇疆:{complete_result.location}, "
                f"ETag:{complete_result.etag}"
            )

        url = getattr(complete_result, "location", None) or ""
        if not url:
            raise RuntimeError("CompleteMultipartUpload 鍝嶅簲涓湭鍖呭惈 location锛屾棤娉曞緱鍒板璞?URL")

        return str(url)

    @overload
    def upload_local_file(
        self,
        local_path: Union[str, Path],
        object_key: str,
        *,
        verbose: bool = True,
    ) -> str: ...

    @overload
    def upload_local_file(
        self,
        local_path: np.ndarray,
        object_key: str,
        *,
        verbose: bool = True,
    ) -> str: ...

    @overload
    def upload_local_file(
        self,
        local_path: PILImageType,
        object_key: str,
        *,
        verbose: bool = True,
    ) -> str: ...

    def upload_local_file(
        self,
        local_path: Union[str, Path, np.ndarray, PILImageType],
        object_key: str,
        *,
        verbose: bool = True,
    ) -> str:
        """
        灏嗘暟鎹潵婧愬垎鐗囦笂浼犺嚦褰撳墠 Bucket 鐨?`object_key`銆?

        `local_path` 鍙负锛?
        - 鏈湴鏂囦欢璺緞
        - OpenCV 甯哥敤鐨?`numpy.ndarray`
        - `PIL.Image.Image`
        """
        if isinstance(local_path, np.ndarray):
            raw = self._encode_cv2_ndarray_to_bytes(local_path)

            def read_part(off: int, n: int) -> bytes:
                return raw[off : off + n]

            return self._multipart_upload_with_reader(
                object_key, len(raw), read_part, verbose=verbose
            )

        if PILImageModule is not None and isinstance(local_path, PILImageModule.Image):
            raw = self._encode_pil_image_to_bytes(local_path)

            def read_part_pil(off: int, n: int) -> bytes:
                return raw[off : off + n]

            return self._multipart_upload_with_reader(
                object_key, len(raw), read_part_pil, verbose=verbose
            )

        path = Path(local_path)  # type: ignore[arg-type]
        if not path.is_file():
            raise FileNotFoundError(f"鏈湴鏂囦欢涓嶅瓨鍦? {path.resolve()}")
        decoded = self._decode_local_image_path(path)
        raw = self._encode_cv2_ndarray_to_bytes(decoded)

        def read_part_file(off: int, n: int) -> bytes:
            return raw[off : off + n]

        return self._multipart_upload_with_reader(
            object_key, len(raw), read_part_file, verbose=verbose
        )

