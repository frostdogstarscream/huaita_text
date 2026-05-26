#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# OSS Python SDK V2锛氬垎鐗囦笂浼犵ず渚嬶紝涓婁紶鎴愬姛鍚庝笅杞藉埌椤圭洰 output/ 鐩綍
# 鍑瘉锛歋taticCredentialsProvider锛堣鍕垮皢瀵嗛挜鎻愪氦鍒扮増鏈簱锛?

import os
from pathlib import Path

import alibabacloud_oss_v2 as oss

from ali.ali_oss_multipart_upload_util import AliOssMultipartUploader

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT = _ROOT / "output"

ACCESS_KEY_ID = "${ALIYUN_ACCESS_KEY_ID}"
ACCESS_KEY_SECRET = "${ALIYUN_ACCESS_KEY_SECRET}"

REGION = "cn-shanghai"
BUCKET = "huaita-person-img"
ENDPOINT = "https://oss-cn-shanghai.aliyuncs.com"
OBJECT_KEY = "person_front/dest.jpg"
LOCAL_FILE_PATH = (
    r"G:\python_pro\huaita_renxiang\resource\person_front\浜哄儚鐓х墖_浣庢竻_缂╂斁绗﹀悎闃块噷杈撳叆.jpg"
)
# 涓嬭浇鐩綍锛汵one 琛ㄧず浣跨敤椤圭洰 output/
DOWNLOAD_DIR = None


def main() -> None:
    uploader = AliOssMultipartUploader(
        access_key_id=ACCESS_KEY_ID,
        access_key_secret=ACCESS_KEY_SECRET,
        bucket=BUCKET,
        region=REGION,
        endpoint=ENDPOINT,
    )
    public_url = uploader.upload_local_file(LOCAL_FILE_PATH, OBJECT_KEY)
    print(f"瀵硅薄 URL: {public_url}")

    client = uploader.client
    bucket = BUCKET
    key = OBJECT_KEY

    download_dir = Path(DOWNLOAD_DIR) if DOWNLOAD_DIR else _DEFAULT_OUTPUT
    download_dir.mkdir(parents=True, exist_ok=True)
    local_download_path = download_dir / Path(key).name

    down_loader = client.downloader()
    dl_result = down_loader.download_file(
        oss.GetObjectRequest(
            bucket=bucket,
            key=key,
        ),
        filepath=str(local_download_path),
    )
    print(f"涓嬭浇瀹屾垚 written: {dl_result.written}, 鏈湴璺緞: {local_download_path.resolve()}")


if __name__ == "__main__":
    main()

