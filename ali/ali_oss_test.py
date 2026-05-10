#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# OSS Python SDK V2：分片上传示例，上传成功后下载到项目 output/ 目录
# 凭证：StaticCredentialsProvider（请勿将密钥提交到版本库）

import os
from pathlib import Path

import alibabacloud_oss_v2 as oss

from ali.ali_oss_multipart_upload_util import AliOssMultipartUploader

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUTPUT = _ROOT / "output"

ACCESS_KEY_ID = os.environ.get("ALI_ACCESS_KEY_ID", "")
ACCESS_KEY_SECRET = os.environ.get("ALI_ACCESS_KEY_SECRET", "")

REGION = "cn-shanghai"
BUCKET = "huaita-person-img"
ENDPOINT = "https://oss-cn-shanghai.aliyuncs.com"
OBJECT_KEY = "person_front/dest.jpg"
LOCAL_FILE_PATH = (
    r"G:\python_pro\huaita_renxiang\resource\person_front\人像照片_低清_缩放符合阿里输入.jpg"
)
# 下载目录；None 表示使用项目 output/
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
    print(f"对象 URL: {public_url}")

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
    print(f"下载完成 written: {dl_result.written}, 本地路径: {local_download_path.resolve()}")


if __name__ == "__main__":
    main()
