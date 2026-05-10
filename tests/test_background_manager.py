"""Test background_manager: items, layout merging, slogan bounds, rotating selection."""

import pytest

from background_manager import (
    get_background_items,
    resolve_person_layout,
    resolve_slogan_bounds,
    resolve_text_layout,
    select_rotating_background,
)


class TestGetBackgroundItems:
    def test_returns_all_items(self, patched_app_state):
        items = get_background_items()
        assert len(items) == 2
        assert items[0]["id"] == "bg_001"
        assert items[1]["id"] == "bg_002"

    def test_items_have_preview_url(self, patched_app_state):
        items = get_background_items()
        assert items[0]["preview_url"].startswith("/static/")

    def test_order_field(self, patched_app_state):
        items = get_background_items()
        assert items[0]["order"] == 1
        assert items[1]["order"] == 2


class TestResolvePersonLayout:
    def test_global_defaults(self, patched_app_state):
        item = {"id": "bg_test"}
        layout = resolve_person_layout(item)
        assert layout["target_height_ratio"] == 0.72
        assert layout["center_x_ratio"] == 0.50

    def test_background_override(self, patched_app_state):
        item = {"id": "bg_test", "person_layout": {"center_x_ratio": 0.3}}
        layout = resolve_person_layout(item)
        assert layout["center_x_ratio"] == 0.3
        assert layout["target_height_ratio"] == 0.72  # global default preserved

    def test_none_layout_handled(self, patched_app_state):
        item = {"id": "bg_test", "person_layout": None}
        layout = resolve_person_layout(item)
        assert layout["target_height_ratio"] == 0.72


class TestResolveTextLayout:
    def test_returns_complete_layout(self, patched_app_state):
        item = {"id": "bg_test"}
        layout = resolve_text_layout(item)
        for key in ["max_lines", "font_size_min", "font_size_max", "line_spacing_min", "line_spacing_max"]:
            assert key in layout

    def test_background_text_layout_overrides(self, patched_app_state):
        item = {
            "id": "bg_test",
            "text_layout": {"max_lines": 2, "font_size_min": 40},
        }
        layout = resolve_text_layout(item)
        assert layout["max_lines"] == 2
        assert layout["font_size_min"] == 40

    def test_font_size_min_never_exceeds_max(self, patched_app_state):
        item = {
            "id": "bg_test",
            "text_layout": {"font_size_min": 200, "font_size_max": 50},
        }
        layout = resolve_text_layout(item)
        assert layout["font_size_min"] <= layout["font_size_max"]


class TestResolveSloganBounds:
    def test_no_region_returns_none(self, patched_app_state):
        layout = {"top_overlay_height": 340}
        result = resolve_slogan_bounds((1080, 1920), layout)
        assert result is None

    def test_valid_region_returns_tuple(self, patched_app_state):
        layout = {
            "text_region": {
                "margin_top_ratio": 0.04,
                "width_ratio": 0.92,
                "height_ratio": 0.21,
            }
        }
        result = resolve_slogan_bounds((1080, 1920), layout)
        assert result is not None
        left, top, width, height = result
        assert width > 0
        assert height > 0

    def test_invalid_ratios_return_none(self, patched_app_state):
        layout = {"text_region": {"width_ratio": 0, "height_ratio": 0, "margin_top_ratio": 0}}
        result = resolve_slogan_bounds((1080, 1920), layout)
        assert result is None


class TestSelectRotatingBackground:
    def test_selects_first_item(self, patched_app_state):
        items = get_background_items()
        bg = select_rotating_background(items)
        assert bg["id"] in {"bg_001", "bg_002"}

    def test_empty_list_raises(self, patched_app_state):
        with pytest.raises(ValueError, match="No background"):
            select_rotating_background([])
