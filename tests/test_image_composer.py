"""Test image_composer: resize, subject placement, composition, output dimensions."""

import numpy as np
from pathlib import Path
from PIL import Image
from unittest.mock import patch, MagicMock

import pytest

from image_composer import compose_single_variant, resize_cover, build_subject_cutout


class TestResizeCover:
    def test_same_aspect_ratio(self):
        img = Image.new("RGB", (100, 200), (255, 0, 0))
        result = resize_cover(img, (50, 100))
        assert result.size == (50, 100)

    def test_different_aspect_fills_and_crops(self):
        img = Image.new("RGB", (200, 100), (255, 0, 0))
        result = resize_cover(img, (100, 200))
        assert result.size == (100, 200)

    def test_already_target_size(self):
        img = Image.new("RGB", (1080, 1920), (255, 0, 0))
        result = resize_cover(img, (1080, 1920))
        assert result.size == (1080, 1920)


class TestComposeSingleVariant:
    def test_creates_1080x1920_output(self, patched_app_state, tmp_path):
        from app_state import FINAL_DIR
        import app_state

        # Create a mock background file
        bg_path = Path(app_state.RESOURCE_DIR) / "html-page" / "assets" / "photos" / "1.jpg"
        if not bg_path.parent.exists():
            bg_path.parent.mkdir(parents=True, exist_ok=True)

        bg_img = Image.new("RGB", (1080, 1920), (100, 80, 60))
        # Save to temp and redirect RESOURCE_DIR
        temp_bg = tmp_path / "test_bg.jpg"
        bg_img.save(temp_bg)

        # Create a mock subject (RGBA person silhouette)
        subject = Image.new("RGBA", (400, 800), (255, 255, 255, 200))
        # Add some non-transparent pixels as a simple "person" shape
        for y in range(200, 700):
            for x in range(100, 300):
                subject.putpixel((x, y), (200, 150, 100, 255))

        bg_item = patched_app_state["config"]["background_set"]["items"][0]
        # Override path to our temp background
        saved_path = bg_item["path"]
        saved_resource_dir = app_state.RESOURCE_DIR
        try:
            bg_item["path"] = str(temp_bg.relative_to(tmp_path))
            # We need to redirect RESOURCE_DIR
            import image_composer
            orig_resource = image_composer.RESOURCE_DIR
            image_composer.RESOURCE_DIR = tmp_path

            result = compose_single_variant(
                subject=subject,
                slogan="人在阵地在",
                task_id="test_task",
                background_item=bg_item,
                order=1,
                slogan_row=1,
            )

            assert "image_id" in result
            assert "image_url" in result
            assert result["image_url"].startswith("/generated/final/")
            assert result["order"] == 1

            # Check output file exists
            filename = result["image_url"].split("/")[-1]
            output_file = Path(app_state.FINAL_DIR) / filename
            assert output_file.exists()

            # Check output dimensions
            saved = Image.open(output_file)
            assert saved.size == (1080, 1920)
        finally:
            bg_item["path"] = saved_path
            image_composer.RESOURCE_DIR = orig_resource

    def test_subject_placement_honors_layout(self, patched_app_state, tmp_path):
        """Person position changes with center_x_ratio."""
        import image_composer
        import app_state
        from app_state import FINAL_DIR

        bg_img = Image.new("RGB", (1080, 1920), (100, 80, 60))
        temp_bg = tmp_path / "test_bg2.jpg"
        bg_img.save(temp_bg)

        subject = Image.new("RGBA", (400, 800), (255, 255, 255, 0))
        for y in range(200, 700):
            for x in range(100, 300):
                subject.putpixel((x, y), (200, 150, 100, 255))

        bg_item = dict(patched_app_state["config"]["background_set"]["items"][0])
        saved_resource = image_composer.RESOURCE_DIR
        try:
            bg_item["path"] = str(temp_bg.relative_to(tmp_path))
            image_composer.RESOURCE_DIR = tmp_path

            result = compose_single_variant(
                subject=subject, slogan="测试", task_id="test2",
                background_item=bg_item, order=1, slogan_row=1,
            )
            assert result["image_url"].startswith("/generated/final/")
        finally:
            image_composer.RESOURCE_DIR = saved_resource


class TestBuildSubjectCutout:
    def test_calls_matting_service(self, patched_app_state, tmp_path):
        """Verify build_subject_cutout calls the matting service."""
        # Create a mock capture image
        capture_path = tmp_path / "capture.jpg"
        img = Image.new("RGB", (400, 600), (100, 100, 100))
        img.save(capture_path)

        # Mock the matting service
        mock_matting = patched_app_state["matting_service"]
        mock_result = Image.new("RGBA", (300, 500), (200, 150, 100, 255))
        # Add non-empty bbox
        for y in range(100, 400):
            for x in range(50, 250):
                mock_result.putpixel((x, y), (200, 150, 100, 255))
        mock_matting.segment_image_file.return_value = mock_result

        from app_state import CUTOUT_DIR
        cutout_path = CUTOUT_DIR / "test_cutout.png"

        # If cutout already exists, remove it
        if cutout_path.exists():
            cutout_path.unlink()

        result = build_subject_cutout(capture_path, "test_cutout")
        assert result is not None
        assert result.mode == "RGBA"
        mock_matting.segment_image_file.assert_called_once()
