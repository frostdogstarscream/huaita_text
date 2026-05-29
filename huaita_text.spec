# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import sys

project_root = Path.cwd().resolve()
python_base = Path(sys.base_prefix).resolve()
conda_bin = python_base / "Library" / "bin"

binaries = []
for dll_name in ("expat.dll", "libexpat.dll"):
    dll_path = conda_bin / dll_name
    if dll_path.is_file():
        binaries.append((str(dll_path), "."))
for site_path in [Path(p) for p in sys.path if p]:
    if site_path.name == "site-packages" and site_path.is_dir():
        for mypyc_path in site_path.glob("*__mypyc*.pyd"):
            binaries.append((str(mypyc_path), "."))

icon_path = project_root / "assets" / "app_icon.ico"

datas = [
    (str(project_root / "html-page"), "html-page"),
    (str(project_root / "config.json"), "."),
    (str(project_root / "fonts"), "fonts"),
    (str(icon_path), "assets"),
    (str(project_root / "ali_seedream" / "ali" / "ali_oss_segment_pipeline_config.yaml"), "ali_seedream/ali"),
    (str(project_root / "ali_seedream_chinamobile" / "ali" / "ali_oss_segment_pipeline_config.yaml"), "ali_seedream_chinamobile/ali"),
]

hiddenimports = [
    "gui_icon",
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
    "package_self_test",
    "ali_seedream.ali",
    "ali_seedream_chinamobile.ali",
    "yaml",
    "charset_normalizer",
    "charset_normalizer.api",
    "charset_normalizer.constant",
    "charset_normalizer.legacy",
    "charset_normalizer.md",
    "charset_normalizer.models",
    "charset_normalizer.utils",
    "charset_normalizer.version",
    "_elementtree",
    "pyexpat",
    "xml.parsers",
    "xml.parsers.expat",
    "xml.etree.ElementTree",
    "xml.dom.expatbuilder",
    "xml.sax.expatreader",
    "defusedxml.common",
    "defusedxml.ElementTree",
    "defusedxml.expatbuilder",
    "defusedxml.expatreader",
    "alibabacloud_oss_v2",
    "alibabacloud_imageseg20191230",
    "alibabacloud_credentials",
    "alibabacloud_tea_openapi",
    "alibabacloud_tea_util",
    "alibabacloud_tea_xml",
    "alibabacloud_openapi_util",
    "alibabacloud_endpoint_util",
    "darabonba.core",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtNetwork",
    "PySide6.QtWidgets",
    "PySide6.QtWebChannel",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
]


a = Analysis(
    ["gui_app.py"],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "IPython",
        "PyQt5",
        "jupyter_client",
        "jupyter_core",
        "matplotlib_inline",
        "matplotlib",
        "paramiko",
        "pkg_resources._vendor",
        "pkg_resources",
        "pytest",
        "setuptools",
        "setuptools._distutils",
        "jedi",
        "ultralytics",
        "torch",
        "torchvision",
        "torchaudio",
        "pandas",
        "pyarrow",
        "scipy",
        "sklearn",
        "numba",
        "llvmlite",
        "dask",
        "distributed",
        "bokeh",
        "notebook",
        "jupyterlab",
        "sphinx",
        "docutils",
        "h5py",
        "tables",
        "openpyxl",
        "sqlalchemy",
        "tkinter",
        "tornado",
        "traitlets",
        "wheel",
        "zmq",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="huaita_text",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(icon_path) if icon_path.is_file() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="huaita_text",
)
