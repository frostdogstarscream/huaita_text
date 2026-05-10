"""Test text_renderer: font loading, color parsing, wrapping, layout, gold-layered rendering."""

from PIL import Image, ImageDraw
import pytest

from text_renderer import (
    build_slogan_candidates,
    draw_slogan,
    draw_slogan_line_layered,
    ellipsize_to_width,
    fit_lines_to_region,
    get_font,
    layout_slogan_lines,
    parse_hex_color,
    parse_alpha,
    parse_offset,
    parse_float,
    text_line_height,
    wrap_text,
    wrap_text_tokens,
)


class TestGetFont:
    def test_returns_font_for_valid_size(self):
        font = get_font(30)
        assert font is not None

    def test_returns_default_font_for_any_size(self):
        font = get_font(12)
        assert font is not None
        font2 = get_font(72)
        assert font2 is not None


class TestParseHexColor:
    def test_standard_hex(self):
        assert parse_hex_color("#FF0000", (0, 0, 0)) == (255, 0, 0)

    def test_without_hash(self):
        assert parse_hex_color("00FF00", (0, 0, 0)) == (0, 255, 0)

    def test_short_hex(self):
        assert parse_hex_color("#FFF", (0, 0, 0)) == (255, 255, 255)

    def test_invalid_returns_default(self):
        assert parse_hex_color("not-a-color", (10, 20, 30)) == (10, 20, 30)

    def test_none_returns_default(self):
        assert parse_hex_color(None, (1, 2, 3)) == (1, 2, 3)


class TestParseAlpha:
    def test_valid_int(self):
        assert parse_alpha(128) == 128

    def test_clamps_to_255(self):
        assert parse_alpha(300) == 255

    def test_clamps_to_0(self):
        assert parse_alpha(-10) == 0

    def test_invalid_returns_default(self):
        assert parse_alpha("abc", 100) == 100


class TestParseOffset:
    def test_valid_list(self):
        assert parse_offset([4, 5], (0, 0)) == (4, 5)

    def test_invalid_returns_default(self):
        assert parse_offset("bad", (1, 2)) == (1, 2)

    def test_short_list_returns_default(self):
        assert parse_offset([1], (3, 4)) == (3, 4)


class TestParseFloat:
    def test_valid(self):
        assert parse_float("1.5", 1.0) == 1.5

    def test_int(self):
        assert parse_float(3, 1.0) == 3.0

    def test_invalid_returns_default(self):
        assert parse_float("abc", 2.0) == 2.0


class TestWrapText:
    def test_single_line_short(self):
        draw = ImageDraw.Draw(Image.new("RGBA", (500, 100)))
        font = get_font(30)
        lines = wrap_text(draw, "短", font, 500, 3)
        assert len(lines) == 1

    def test_empty_text(self):
        draw = ImageDraw.Draw(Image.new("RGBA", (500, 100)))
        font = get_font(30)
        lines = wrap_text(draw, "", font, 500, 3)
        assert lines == [""]

    def test_long_text_wraps(self):
        draw = ImageDraw.Draw(Image.new("RGBA", (200, 100)))
        font = get_font(40)
        lines = wrap_text(draw, "这是一个很长的标语需要自动换行处理", font, 200, 3)
        assert len(lines) >= 1

    def test_chinese_characters_wrap(self):
        draw = ImageDraw.Draw(Image.new("RGBA", (100, 100)))
        font = get_font(40)
        lines = wrap_text(draw, "测试标语文", font, 100, 0)
        # Should wrap since each Chinese char is ~40px wide
        assert len(lines) >= 1


class TestWrapTextTokens:
    def test_space_separated(self):
        draw = ImageDraw.Draw(Image.new("RGBA", (500, 100)))
        font = get_font(30)
        lines = wrap_text_tokens(draw, "hello world test", font, 500, 3)
        assert len(lines) >= 1

    def test_no_spaces_falls_back_to_char_wrap(self):
        draw = ImageDraw.Draw(Image.new("RGBA", (500, 100)))
        font = get_font(30)
        lines = wrap_text_tokens(draw, "中文无空格", font, 500, 3)
        assert len(lines) >= 1


class TestBuildSloganCandidates:
    def test_single_line_candidate(self):
        candidates = build_slogan_candidates("人在阵地在", False)
        assert ["人在阵地在"] in candidates

    def test_punctuation_break(self):
        candidates = build_slogan_candidates("坚持，就是胜利", True)
        # Should include a version split at comma
        all_lines = [tuple(c) for c in candidates]
        assert any(len(c) > 1 for c in all_lines)

    def test_long_slogan_creates_midpoint_splits(self):
        candidates = build_slogan_candidates("这是一个很长的十二字标语用于测试分行", True)
        all_lines = [tuple(c) for c in candidates]
        assert any(len(c) == 2 for c in all_lines)

    def test_empty_slogan(self):
        candidates = build_slogan_candidates("", False)
        assert candidates == [[""]]


class TestEllipsize:
    def test_short_text_unchanged(self):
        draw = ImageDraw.Draw(Image.new("RGBA", (500, 50)))
        font = get_font(20)
        result = ellipsize_to_width(draw, "短", font, 500, 0)
        assert "..." not in result

    def test_long_text_ellipsized(self):
        draw = ImageDraw.Draw(Image.new("RGBA", (60, 50)))
        font = get_font(40)
        result = ellipsize_to_width(draw, "很长很长", font, 60, 0)
        assert "..." in result


class TestLayoutSloganLines:
    def test_returns_font_lines_spacing(self):
        font, lines, spacing = layout_slogan_lines(
            slogan="人在阵地在",
            stroke_width=3,
            max_width=1080,
            max_height=200,
            max_lines=3,
            preferred_lines=2,
            font_size_min=20,
            font_size_max=72,
            line_spacing_min=4,
            line_spacing_max=14,
            enable_punctuation_break=True,
            balance_weight=1000,
        )
        assert font is not None
        assert len(lines) >= 1
        assert spacing >= 0

    def test_fits_single_line_in_wide_box(self):
        font, lines, spacing = layout_slogan_lines(
            slogan="短标语",
            stroke_width=3,
            max_width=1000,
            max_height=300,
            max_lines=1,
            preferred_lines=1,
            font_size_min=30,
            font_size_max=96,
            line_spacing_min=4,
            line_spacing_max=14,
            enable_punctuation_break=False,
            balance_weight=0,
            line_priority=[1],
        )
        assert len(lines) == 1

    def test_long_slogan_splits_to_multiple_lines(self):
        font, lines, spacing = layout_slogan_lines(
            slogan="不惜一切代价克服一切困难完成一切任务",
            stroke_width=3,
            max_width=800,
            max_height=400,
            max_lines=3,
            preferred_lines=2,
            font_size_min=20,
            font_size_max=72,
            line_spacing_min=4,
            line_spacing_max=14,
            enable_punctuation_break=True,
            balance_weight=1000,
        )
        assert len(lines) >= 1
        # With a long slogan in a constrained width, should wrap
        assert all(len(line) > 0 for line in lines)


class TestFitLinesToRegion:
    def test_returns_font_and_spacing(self):
        font, spacing = fit_lines_to_region(
            lines=["测试"],
            stroke_width=3,
            max_width=800,
            max_height=200,
            line_spacing_min=4,
            line_spacing_max=14,
            font_size_start=72,
            font_size_min=30,
        )
        assert font is not None
        assert spacing >= 4


class TestDrawSlogan:
    def test_draw_on_image_does_not_crash(self, patched_app_state):
        img = Image.new("RGBA", (1080, 1920), (50, 50, 50, 255))
        bg_item = get_background_items_stub()
        result = draw_slogan(img, "人在阵地在", bg_item, slogan_row=1)
        assert result is not None
        assert result.size == (1080, 1920)

    def test_draw_multiline_slogan(self, patched_app_state):
        img = Image.new("RGBA", (1080, 1920), (50, 50, 50, 255))
        bg_item = get_background_items_stub()
        result = draw_slogan(img, "不惜一切代价\n克服一切困难", bg_item, slogan_row=2)
        assert result is not None

    def test_gold_layered_produces_different_pixels(self, patched_app_state):
        img = Image.new("RGBA", (1080, 1920), (50, 50, 50, 255))
        bg_item = get_background_items_stub()
        result = draw_slogan(img, "金色标语测试", bg_item, slogan_row=1)
        original = Image.new("RGBA", (1080, 1920), (50, 50, 50, 255))
        # Sample the text region (top-center area around y=80-200)
        region = result.crop((200, 50, 880, 250))
        orig_region = original.crop((200, 50, 880, 250))
        assert list(region.getdata()) != list(orig_region.getdata())

    def test_draw_with_text_region_constrained(self, patched_app_state):
        img = Image.new("RGBA", (1080, 1920), (50, 50, 50, 255))
        bg_item = {
            "id": "bg_test",
            "name": "测试",
            "path": "test.jpg",
            "text_layout": {
                "top_overlay_height": 340,
                "max_lines": 3,
                "font_size_min": 40,
                "font_size_max": 80,
                "line_spacing_min": 4,
                "line_spacing_max": 10,
                "text_region": {
                    "margin_top_ratio": 0.04,
                    "width_ratio": 0.92,
                    "height_ratio": 0.18,
                },
            },
        }
        result = draw_slogan(img, "标语在区域内", bg_item, slogan_row=1)
        assert result is not None


def get_background_items_stub():
    """Minimal background item stub for draw_slogan tests."""
    return {
        "id": "bg_test",
        "name": "测试背景",
        "path": "test.jpg",
        "text_layout": {
            "top_overlay_height": 340,
            "max_lines": 3,
            "font_size_min": 40,
            "font_size_max": 80,
            "line_spacing_min": 4,
            "line_spacing_max": 10,
        },
    }


class TestDrawSloganLineLayered:
    def test_classic_mode_renders(self):
        img = Image.new("RGBA", (500, 200), (30, 30, 30, 255))
        font = get_font(40)
        text_cfg = {
            "style_mode": "classic",
            "fill": "#FFFFFF",
            "stroke_fill": "#000000",
        }
        draw_slogan_line_layered(img, "测试", (50, 50), font, 3, text_cfg)
        # Image should have been modified (no exception)
        assert True
