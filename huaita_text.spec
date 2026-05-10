# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


project_root = Path.cwd().resolve()

datas = [
    (str(project_root / "html-page"), "html-page"),
    (str(project_root / "config.json"), "."),
]

hiddenimports = [
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
]


a = Analysis(
    ["run_app.py"],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "IPython",
        "PyQt5",
        "cryptography",
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
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
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
