"""Test runtime_paths: dev mode vs frozen/PyInstaller path resolution."""

import sys
from pathlib import Path
from unittest.mock import patch

from runtime_paths import get_app_paths, get_resource_base_dir, get_runtime_base_dir


class TestDevMode:
    def test_get_runtime_base_dir_returns_repo_root(self):
        base = get_runtime_base_dir()
        assert isinstance(base, Path)
        assert (base / "main.py").exists()

    def test_get_resource_base_dir_same_as_runtime_in_dev(self):
        assert get_resource_base_dir() == get_runtime_base_dir()

    def test_get_app_paths_returns_all_keys(self):
        paths = get_app_paths()
        for key in ["base_dir", "resource_dir", "frontend_dir", "config_path",
                     "output_dir", "capture_dir", "cutout_dir", "final_dir", "fonts_dir"]:
            assert key in paths
            assert isinstance(paths[key], Path)

    def test_frontend_dir_points_to_html_page(self):
        paths = get_app_paths()
        assert paths["frontend_dir"].name == "html-page"

    def test_config_path_is_config_json(self):
        paths = get_app_paths()
        assert paths["config_path"].name == "config.json"


class TestFrozenMode:
    def test_frozen_base_dir_is_exe_parent(self):
        with patch.object(sys, "frozen", True, create=True):
            with patch.object(sys, "executable", r"C:\deploy\huaita_text\app.exe"):
                base = get_runtime_base_dir()
                assert str(base) == str(Path(r"C:\deploy\huaita_text"))

    def test_frozen_resource_dir_same_as_base(self):
        with patch.object(sys, "frozen", True, create=True):
            with patch.object(sys, "executable", r"C:\deploy\huaita_text\app.exe"):
                assert get_resource_base_dir() == get_runtime_base_dir()

    def test_frozen_app_paths_consistent(self):
        with patch.object(sys, "frozen", True, create=True):
            with patch.object(sys, "executable", r"D:\app\myapp.exe"):
                paths = get_app_paths()
                assert paths["base_dir"] == Path(r"D:\app")
                assert paths["resource_dir"] == Path(r"D:\app")
                assert paths["config_path"] == Path(r"D:\app\config.json")
                assert paths["frontend_dir"] == Path(r"D:\app\html-page")
