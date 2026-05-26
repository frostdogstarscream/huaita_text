# -*- coding: utf-8 -*-
"""
闃块噷浜?OSS 鍒嗙墖涓婁紶宸ュ叿锛氭湰鍦拌矾寰?/ OpenCV ndarray / PIL Image 鈫?multipart upload 鈫?杩斿洖瀵硅薄鍏綉 URL銆?
"""
from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import Callable, TYPE_CHECKING, Union, overload

import alibabacloud_oss_v2 as oss
import cv2
import numpy as np

if TYPE_CHECKING:
    from PIL.Image import Image as PILImageType

try:
    from PIL import Image as PILImageModule
except ImportError:
    PILImageModule = None  # type: ignore[misc, assignment]

# 涓?`ali_oss_test.py` 榛樿閰嶇疆瀵归綈锛涙瀯閫犳椂鍙鐩?
DEFAULT_ACCESS_KEY_ID = "${ALIYUN_ACCESS_KEY_ID}"
DEFAULT_ACCESS_KEY_SECRET = "${ALIYUN_ACCESS_KEY_SECRET}"
DEFAULT_BUCKET = "huaita-person-img"
DEFAULT_REGION = "cn-shanghai"
DEFAULT_ENDPOINT = "https://oss-cn-shanghai.aliyuncs.com"
DEFAULT_PART_SIZE = 100 * 1024


class AliOssMultipartUploader:
    """鏈湴璺緞鎴栧唴瀛樺浘鍍忓垎鐗囦笂浼犺嚦 OSS锛屾棩蹇楁牸寮忎笌 `ali_oss_test` 绀轰緥涓€鑷淬€?""

    def __init__(
        self,
        access_key_id: str = DEFAULT_ACCESS_KEY_ID,
        access_key_secret: str = DEFAULT_ACCESS_KEY_SECRET,
        bucket: str = DEFAULT_BUCKET,
        region: str = DEFAULT_REGION,
        endpoint: str = DEFAULT_ENDPOINT,
        part_size: int = DEFAULT_PART_SIZE,
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

    @property
    def client(self) -> oss.Client:
        return self._client

    @property
    def bucket(self) -> str:
        return self._bucket

    @staticmethod
    def _encode_cv2_ndarray_to_bytes(arr: np.ndarray) -> bytes:
        if not isinstance(arr, np.ndarray):
            raise TypeError("搴斾负 numpy.ndarray")
        img = arr
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
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

    @staticmethod
    def _encode_pil_image_to_bytes(im: PILImageType) -> bytes:
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

        ``local_path`` 鍙负锛?
        - 鏈湴鏂囦欢璺緞锛坄`str`` / ``Path``锛?
        - OpenCV 甯哥敤鐨?``numpy.ndarray``锛圔GR / BGRA / 鐏板害锛寀int8锛涚紪鐮佷负 JPG 鎴?PNG锛?
        - ``PIL.Image.Image``锛堜緷璧?Pillow锛涚紪鐮佷负 JPEG 鎴?PNG锛?

        Returns:
            涓婁紶鎴愬姛鍚庣殑瀵硅薄 URL锛圫DK CompleteMultipartUpload 杩斿洖鐨?location锛夈€?
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
        file_path = str(path)
        file_size = os.path.getsize(file_path)

        with open(file_path, "rb") as f:

            def read_part_file(off: int, n: int) -> bytes:
                f.seek(off)
                return f.read(n)

            return self._multipart_upload_with_reader(
                object_key, file_size, read_part_file, verbose=verbose
            )

