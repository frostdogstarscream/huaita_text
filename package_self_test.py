"""Frozen package self-test for imports that PyInstaller can miss."""

from __future__ import annotations

import importlib
import traceback
from typing import Callable


def _check_import(module_name: str) -> None:
    importlib.import_module(module_name)


def _check_xml_expat() -> None:
    import pyexpat  # noqa: F401
    from xml.dom import expatbuilder  # noqa: F401
    from xml.etree import ElementTree
    from xml.parsers import expat

    parser = expat.ParserCreate()
    parser.Parse(b"<root><item>ok</item></root>", True)
    ElementTree.XML("<root><item>ok</item></root>")
    ElementTree.XMLParser()


def _check_cv2_pillow_yaml() -> None:
    import cv2
    import numpy as np
    import yaml
    from PIL import Image

    arr = np.zeros((2, 2, 3), dtype=np.uint8)
    ok, _ = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    Image.fromarray(arr)
    yaml.safe_load("ok: true")


def _check_requests_charset() -> None:
    import charset_normalizer
    import requests

    version = getattr(charset_normalizer, "__version__", "")
    if not version:
        raise RuntimeError("charset_normalizer.__version__ is empty")
    requests.utils.get_encoding_from_headers({"content-type": "text/plain; charset=utf-8"})


def _check_pyside6_webengine() -> None:
    from PySide6 import QtCore, QtWidgets  # noqa: F401
    from PySide6.QtWebEngineCore import QWebEngineSettings  # noqa: F401
    from PySide6.QtWebEngineWidgets import QWebEngineView  # noqa: F401


def _check_project_pipeline_imports() -> None:
    from ali import AliOssSegmentPipeline as DefaultPipeline  # noqa: F401
    from ali_segment_service import AliSegmentService  # noqa: F401
    from ali_seedream_chinamobile.ali import AliOssSegmentPipeline as SuxiaobanPipeline  # noqa: F401
    from ali_seedream_chinamobile.ali.suxiaobanengine_segment_greenscreen import (
        SuxiaobanImageGenerationsClient,  # noqa: F401
    )


def run_self_test() -> int:
    checks: list[tuple[str, Callable[[], None]]] = [
        ("xml-expat", _check_xml_expat),
        ("cv2-pillow-yaml", _check_cv2_pillow_yaml),
        ("fastapi", lambda: _check_import("fastapi")),
        ("uvicorn", lambda: _check_import("uvicorn")),
        ("requests-charset", _check_requests_charset),
        ("pyside6-webengine", _check_pyside6_webengine),
        ("serial", lambda: _check_import("serial")),
        ("alibabacloud_oss_v2", lambda: _check_import("alibabacloud_oss_v2")),
        ("alibabacloud_imageseg20191230", lambda: _check_import("alibabacloud_imageseg20191230")),
        ("alibabacloud_credentials", lambda: _check_import("alibabacloud_credentials")),
        ("alibabacloud_tea_openapi", lambda: _check_import("alibabacloud_tea_openapi")),
        ("alibabacloud_tea_util", lambda: _check_import("alibabacloud_tea_util")),
        ("alibabacloud_tea_xml", lambda: _check_import("alibabacloud_tea_xml")),
        ("alibabacloud_openapi_util", lambda: _check_import("alibabacloud_openapi_util")),
        ("alibabacloud_endpoint_util", lambda: _check_import("alibabacloud_endpoint_util")),
        ("darabonba.core", lambda: _check_import("darabonba.core")),
        ("project-pipeline-imports", _check_project_pipeline_imports),
    ]

    print("[SELF-TEST] start")
    for name, check in checks:
        try:
            check()
        except Exception:
            print(f"[SELF-TEST] FAIL {name}")
            traceback.print_exc()
            return 1
        print(f"[SELF-TEST] OK {name}")
    print("[SELF-TEST] all checks passed")
    return 0
