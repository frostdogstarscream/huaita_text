"""Text rendering: font loading, color parsing, line layout, and slogan drawing."""

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageFilter

from app_state import APP_PATHS, APP_STATE
from background_manager import resolve_slogan_bounds, resolve_text_layout
from config_manager import deep_merge, normalize_mojibake_text
from slogan_manager import (
    normalize_slogan_lines_to_row_count,
    resolve_text_tuning,
    slogan_explicit_lines,
)


def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        APP_PATHS["fonts_dir"] / "default.ttf",
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyhbd.ttc"),
        Path("C:/Windows/Fonts/msyh.ttf"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def parse_hex_color(value: str | None, default: tuple[int, int, int]) -> tuple[int, int, int]:
    if not isinstance(value, str):
        return default
    text = value.strip()
    if text.startswith("#"):
        text = text[1:]
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) != 6:
        return default
    try:
        return tuple(int(text[idx : idx + 2], 16) for idx in (0, 2, 4))  # type: ignore[return-value]
    except ValueError:
        return default


def parse_alpha(value: Any, default: int = 255) -> int:
    try:
        alpha = int(value)
    except (TypeError, ValueError):
        alpha = default
    return max(0, min(alpha, 255))


def parse_offset(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return default
    return default


def parse_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def build_vertical_gradient(size: tuple[int, int], palette: list[tuple[int, int, int]]) -> Image.Image:
    width, height = size
    if width <= 0 or height <= 0:
        return Image.new("RGBA", (1, 1), (255, 255, 255, 255))
    if not palette:
        palette = [(255, 255, 255)]
    if len(palette) == 1:
        return Image.new("RGBA", size, (*palette[0], 255))

    gradient = Image.new("RGBA", size, (255, 255, 255, 255))
    px = gradient.load()
    segments = len(palette) - 1
    for y in range(height):
        pos = (y / max(height - 1, 1)) * segments
        idx = min(int(pos), segments - 1)
        frac = pos - idx
        c1 = palette[idx]
        c2 = palette[idx + 1]
        row_color = (
            int(c1[0] + (c2[0] - c1[0]) * frac),
            int(c1[1] + (c2[1] - c1[1]) * frac),
            int(c1[2] + (c2[2] - c1[2]) * frac),
            255,
        )
        for x in range(width):
            px[x, y] = row_color
    return gradient


def draw_slogan_line_layered(
    image: Image.Image,
    line: str,
    position: tuple[int, int],
    font: ImageFont.ImageFont,
    stroke_width: int,
    text_cfg: dict[str, Any],
) -> None:
    x, y = position
    draw = ImageDraw.Draw(image)
    style_mode = str(text_cfg.get("style_mode", "classic")).lower()
    horizontal_scale = max(0.72, min(parse_float(text_cfg.get("horizontal_scale", 1.0), 1.0), 1.2))
    if style_mode != "gold_layered":
        base_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        base_draw = ImageDraw.Draw(base_layer)
        base_draw.text(
            (x, y),
            line,
            font=font,
            fill=text_cfg["fill"],
            stroke_width=stroke_width,
            stroke_fill=text_cfg["stroke_fill"],
        )
        if abs(horizontal_scale - 1.0) < 0.01:
            image.alpha_composite(base_layer)
            return
        base_bbox = base_layer.getbbox()
        if not base_bbox:
            return
        left, top, right, bottom = base_bbox
        segment = base_layer.crop(base_bbox)
        scaled_w = max(int((right - left) * horizontal_scale), 1)
        segment = segment.resize((scaled_w, bottom - top), Image.LANCZOS)
        target_x = int((left + right - scaled_w) / 2)
        image.alpha_composite(segment, dest=(target_x, top))
        return

    outline_color = parse_hex_color(text_cfg.get("outline_dark"), (90, 51, 5))
    shadow_color = parse_hex_color(text_cfg.get("shadow_color"), (58, 29, 6))
    shadow_offset = parse_offset(text_cfg.get("shadow_offset"), (2, 3))
    shadow_blur = max(int(text_cfg.get("shadow_blur", 2)), 0)
    shadow_alpha = parse_alpha(text_cfg.get("shadow_alpha", 180), 180)
    highlight_color = parse_hex_color(text_cfg.get("highlight_color"), (255, 248, 214))
    highlight_offset = parse_offset(text_cfg.get("highlight_offset"), (0, -1))
    highlight_alpha = parse_alpha(text_cfg.get("highlight_alpha", 92), 92)
    inner_glow_color = parse_hex_color(text_cfg.get("inner_glow_color"), (255, 243, 184))
    inner_glow_alpha = parse_alpha(text_cfg.get("inner_glow_alpha", 58), 58)
    specular_color = parse_hex_color(text_cfg.get("specular_color"), (255, 254, 244))
    specular_strength = parse_alpha(text_cfg.get("specular_strength", 188), 188)
    specular_band_top_ratio = max(0.0, min(parse_float(text_cfg.get("specular_band_top_ratio", 0.22), 0.22), 1.0))
    specular_band_height_ratio = max(
        0.04,
        min(parse_float(text_cfg.get("specular_band_height_ratio", 0.18), 0.18), 0.6),
    )
    outline_width = max(int(text_cfg.get("outline_width", 2)), 0)
    palette_values = text_cfg.get("gold_palette") or []
    palette = [parse_hex_color(item, (216, 169, 65)) for item in palette_values if isinstance(item, str)]
    if not palette:
        palette = [(253, 244, 192), (216, 169, 65), (143, 90, 20)]

    line_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    line_draw = ImageDraw.Draw(line_layer)

    if shadow_blur > 0 or shadow_offset != (0, 0):
        shadow_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow_layer)
        shadow_rgba = (*shadow_color, shadow_alpha)
        shadow_draw.text(
            (x + shadow_offset[0], y + shadow_offset[1]),
            line,
            font=font,
            fill=shadow_rgba,
            stroke_width=max(stroke_width + outline_width, stroke_width),
            stroke_fill=shadow_rgba,
        )
        if shadow_blur > 0:
            shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=shadow_blur))
        line_layer.alpha_composite(shadow_layer)

    dark_rgba = (*outline_color, 255)
    line_draw.text(
        (x, y),
        line,
        font=font,
        fill=dark_rgba,
        stroke_width=max(stroke_width + outline_width, stroke_width),
        stroke_fill=dark_rgba,
    )

    bbox = draw.textbbox((x, y), line, font=font, stroke_width=max(stroke_width, 1))
    text_w = max(bbox[2] - bbox[0], 1)
    text_h = max(bbox[3] - bbox[1], 1)
    text_mask = Image.new("L", (text_w, text_h), 0)
    mask_draw = ImageDraw.Draw(text_mask)
    mask_draw.text(
        (-bbox[0], -bbox[1]),
        line,
        font=font,
        fill=255,
        stroke_width=max(stroke_width, 1),
        stroke_fill=255,
    )
    gradient = build_vertical_gradient((text_w, text_h), palette)
    line_layer.paste(gradient, (bbox[0], bbox[1]), text_mask)

    if inner_glow_alpha > 0:
        inner_glow = Image.new("RGBA", (text_w, text_h), (*inner_glow_color, inner_glow_alpha))
        inner_mask = text_mask.filter(ImageFilter.GaussianBlur(radius=0.9))
        line_layer.paste(inner_glow, (bbox[0], bbox[1]), inner_mask)

    if specular_strength > 0:
        band_layer = Image.new("RGBA", (text_w, text_h), (0, 0, 0, 0))
        band_draw = ImageDraw.Draw(band_layer)
        top = int(text_h * specular_band_top_ratio)
        band_h = max(int(text_h * specular_band_height_ratio), 2)
        bottom = min(top + band_h, text_h)
        for row in range(top, bottom):
            t = (row - top) / max(bottom - top - 1, 1)
            alpha = int(specular_strength * (1.0 - abs(2 * t - 1)))
            band_draw.line([(0, row), (text_w, row)], fill=(*specular_color, alpha))
        soft_band = band_layer.filter(ImageFilter.GaussianBlur(radius=0.7))
        line_layer.paste(soft_band, (bbox[0], bbox[1]), text_mask)

    highlight_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    highlight_draw = ImageDraw.Draw(highlight_layer)
    highlight_rgba = (*highlight_color, highlight_alpha)
    highlight_draw.text(
        (x + highlight_offset[0], y + highlight_offset[1]),
        line,
        font=font,
        fill=highlight_rgba,
        stroke_width=max(stroke_width - 1, 0),
        stroke_fill=highlight_rgba,
    )
    line_layer.alpha_composite(highlight_layer)

    if abs(horizontal_scale - 1.0) < 0.01:
        image.alpha_composite(line_layer)
        return
    line_bbox = line_layer.getbbox()
    if not line_bbox:
        return
    left, top, right, bottom = line_bbox
    segment = line_layer.crop(line_bbox)
    scaled_w = max(int((right - left) * horizontal_scale), 1)
    segment = segment.resize((scaled_w, bottom - top), Image.LANCZOS)
    target_x = int((left + right - scaled_w) / 2)
    image.alpha_composite(segment, dest=(target_x, top))


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    stroke_width: int,
) -> list[str]:
    if not text:
        return [""]

    lines: list[str] = []
    current = ""
    for char in text:
        candidate = current + char
        bbox = draw.textbbox((0, 0), candidate, font=font, stroke_width=stroke_width)
        width = bbox[2] - bbox[0]
        if current and width > max_width:
            lines.append(current)
            current = char
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def wrap_text_tokens(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    stroke_width: int,
) -> list[str]:
    chunks = [chunk for chunk in text.split(" ") if chunk]
    if not chunks:
        return wrap_text(draw, text, font, max_width, stroke_width)
    lines: list[str] = []
    current = ""
    for chunk in chunks:
        candidate = chunk if not current else f"{current} {chunk}"
        bbox = draw.textbbox((0, 0), candidate, font=font, stroke_width=stroke_width)
        width = bbox[2] - bbox[0]
        if current and width > max_width:
            lines.append(current)
            current = chunk
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def build_slogan_candidates(slogan: str, enable_punctuation_break: bool) -> list[list[str]]:
    slogan = slogan.strip()
    if not slogan:
        return [[""]]

    candidates: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()

    def add_candidate(lines: list[str]) -> None:
        cleaned = [line.strip() for line in lines if line.strip()]
        if not cleaned:
            cleaned = [""]
        key = tuple(cleaned)
        if key not in seen:
            seen.add(key)
            candidates.append(cleaned)

    add_candidate([slogan])

    if " " in slogan:
        add_candidate([part.strip() for part in slogan.split(" ") if part.strip()])

    if enable_punctuation_break:
        punctuations = "，。！？；：、,.!?;:"
        chars = []
        for ch in slogan:
            chars.append(ch)
            if ch in punctuations:
                chars.append("\n")
        punct_split = "".join(chars)
        add_candidate([part.strip() for part in punct_split.split("\n") if part.strip()])

    plain = slogan.replace(" ", "")
    if len(plain) >= 8:
        middle = len(plain) // 2
        for offset in range(0, 4):
            for idx in {middle - offset, middle + offset}:
                if 2 <= idx <= len(plain) - 2:
                    add_candidate([plain[:idx], plain[idx:]])

    return candidates


def text_line_height(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, stroke_width: int) -> int:
    bbox = draw.textbbox((0, 0), "国", font=font, stroke_width=stroke_width)
    return max(bbox[3] - bbox[1], 1)


def ellipsize_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    stroke_width: int,
) -> str:
    ellipsis = "..."
    if not text:
        return ellipsis
    if draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)[2] <= max_width:
        return text
    trimmed = text
    while trimmed:
        candidate = trimmed + ellipsis
        bbox = draw.textbbox((0, 0), candidate, font=font, stroke_width=stroke_width)
        width = bbox[2] - bbox[0]
        if width <= max_width:
            return candidate
        trimmed = trimmed[:-1]
    return ellipsis


def layout_slogan_lines_fixed(
    lines: list[str],
    stroke_width: int,
    max_width: int,
    max_height: int,
    max_lines: int,
    font_size_min: int,
    font_size_max: int,
    line_spacing_min: int,
    line_spacing_max: int,
) -> tuple[ImageFont.ImageFont, list[str], int]:
    """Fit explicit slogan lines (from config line breaks) without re-wrapping."""
    cleaned = [ln.strip() for ln in lines if ln.strip()]
    if not cleaned:
        cleaned = [""]
    cleaned = cleaned[:max_lines]
    measure_draw = ImageDraw.Draw(Image.new("RGBA", (16, 16)))
    fallback_font = get_font(font_size_min)
    fallback_spacing = line_spacing_min
    fallback_lines = list(cleaned)
    best_total_h = 10**9

    for font_size in range(font_size_max, font_size_min - 1, -1):
        font = get_font(font_size)
        ratio = 1.0
        if font_size_max > font_size_min:
            ratio = (font_size - font_size_min) / (font_size_max - font_size_min)
        spacing = int(line_spacing_min + (line_spacing_max - line_spacing_min) * ratio)
        spacing = max(spacing, line_spacing_min)
        fitted: list[str] = []
        heights: list[int] = []
        for seg in cleaned:
            line = seg
            bbox0 = measure_draw.textbbox((0, 0), line, font=font, stroke_width=stroke_width)
            if bbox0[2] - bbox0[0] > max_width:
                line = ellipsize_to_width(measure_draw, line, font, max_width, stroke_width)
            bbox = measure_draw.textbbox((0, 0), line, font=font, stroke_width=stroke_width)
            fitted.append(line)
            heights.append(bbox[3] - bbox[1])
        total_h = sum(heights) + max(len(fitted) - 1, 0) * spacing
        if total_h <= max_height:
            return font, fitted, spacing
        if total_h < best_total_h:
            best_total_h = total_h
            fallback_font = font
            fallback_lines = fitted
            fallback_spacing = spacing
    return fallback_font, fallback_lines, fallback_spacing


def layout_slogan_lines(
    slogan: str,
    stroke_width: int,
    max_width: int,
    max_height: int,
    max_lines: int,
    preferred_lines: int,
    font_size_min: int,
    font_size_max: int,
    line_spacing_min: int,
    line_spacing_max: int,
    enable_punctuation_break: bool,
    balance_weight: int,
    line_priority: list[int] | None = None,
) -> tuple[ImageFont.ImageFont, list[str], int]:
    measure_draw = ImageDraw.Draw(Image.new("RGBA", (16, 16)))
    preferred_lines = max(min(preferred_lines, max_lines), 1)
    priorities = [preferred_lines, 1, 2, 3] if not line_priority else [max(int(v), 1) for v in line_priority]
    priorities = [line for line in priorities if line <= max_lines]
    if not priorities:
        priorities = [min(max_lines, 3), 1][:1]

    fallback_font = get_font(font_size_min)
    fallback_lines = [slogan] if slogan else [""]
    fallback_spacing = line_spacing_min
    fallback_line_h = text_line_height(measure_draw, fallback_font, stroke_width)
    fallback_total_h = len(fallback_lines) * fallback_line_h + max(len(fallback_lines) - 1, 0) * fallback_spacing
    best_by_line_count: dict[int, tuple[tuple[int, int], ImageFont.ImageFont, list[str], int]] = {}

    for font_size in range(font_size_max, font_size_min - 1, -1):
        font = get_font(font_size)
        ratio = 1.0
        if font_size_max > font_size_min:
            ratio = (font_size - font_size_min) / (font_size_max - font_size_min)
        spacing = int(line_spacing_min + (line_spacing_max - line_spacing_min) * ratio)
        spacing = max(spacing, line_spacing_min)
        auto_candidates = [
            wrap_text(measure_draw, slogan, font, max_width, stroke_width),
            wrap_text_tokens(measure_draw, slogan, font, max_width, stroke_width),
        ]
        explicit_candidates = build_slogan_candidates(slogan, enable_punctuation_break)
        for explicit in explicit_candidates:
            explicit_fit: list[str] = []
            for segment in explicit:
                explicit_fit.extend(wrap_text(measure_draw, segment, font, max_width, stroke_width))
            auto_candidates.append(explicit_fit)

        dedup: set[tuple[str, ...]] = set()
        for lines in auto_candidates:
            key = tuple(lines)
            if key in dedup:
                continue
            dedup.add(key)

            line_count = len(lines)
            line_h = text_line_height(measure_draw, font, stroke_width)
            total_h = line_count * line_h + max(line_count - 1, 0) * spacing

            if line_count > max_lines:
                if total_h < fallback_total_h:
                    fallback_font = font
                    fallback_lines = lines
                    fallback_spacing = spacing
                    fallback_total_h = total_h
                continue

            widths: list[int] = []
            for line in lines:
                bbox = measure_draw.textbbox((0, 0), line, font=font, stroke_width=stroke_width)
                widths.append(bbox[2] - bbox[0])
            width_balance = (max(widths) - min(widths)) if len(widths) > 1 else 0

            if total_h <= max_height:
                score = (width_balance * balance_weight, -font_size)
                current = best_by_line_count.get(line_count)
                if current is None or score < current[0]:
                    best_by_line_count[line_count] = (score, font, lines, spacing)
                continue

            overflow = total_h - max_height
            if overflow < (fallback_total_h - max_height):
                fallback_font = font
                fallback_lines = lines
                fallback_spacing = spacing
                fallback_total_h = total_h

    for target_lines in priorities:
        candidate = best_by_line_count.get(target_lines)
        if candidate is not None:
            return candidate[1], candidate[2], candidate[3]

    if len(fallback_lines) > max_lines:
        fallback_lines = fallback_lines[:max_lines]
    if fallback_lines:
        fallback_lines[-1] = ellipsize_to_width(
            measure_draw,
            fallback_lines[-1],
            fallback_font,
            max_width,
            stroke_width,
        )
    return fallback_font, fallback_lines, fallback_spacing


def fit_lines_to_region(
    lines: list[str],
    stroke_width: int,
    max_width: int,
    max_height: int,
    line_spacing_min: int,
    line_spacing_max: int,
    font_size_start: int,
    font_size_min: int,
) -> tuple[ImageFont.ImageFont, int]:
    """Final safety pass: reduce font size until all lines fit box."""
    measure_draw = ImageDraw.Draw(Image.new("RGBA", (16, 16)))
    start = max(font_size_start, font_size_min)
    fallback_font = get_font(font_size_min)
    fallback_spacing = max(line_spacing_min, 0)
    for font_size in range(start, font_size_min - 1, -1):
        font = get_font(font_size)
        ratio = 1.0
        if start > font_size_min:
            ratio = (font_size - font_size_min) / (start - font_size_min)
        spacing = int(line_spacing_min + (line_spacing_max - line_spacing_min) * ratio)
        spacing = max(spacing, line_spacing_min)
        widths: list[int] = []
        heights: list[int] = []
        for line in lines:
            bbox = measure_draw.textbbox((0, 0), line, font=font, stroke_width=stroke_width)
            widths.append(bbox[2] - bbox[0])
            heights.append(bbox[3] - bbox[1])
        total_h = sum(heights) + max(len(lines) - 1, 0) * spacing
        if widths and max(widths) <= max_width and total_h <= max_height:
            return font, spacing
        fallback_font = font
        fallback_spacing = spacing
    return fallback_font, fallback_spacing


def draw_slogan(
    image: Image.Image,
    slogan: str,
    background_item: dict[str, Any],
    slogan_row: int | None = None,
) -> Image.Image:
    cfg = APP_STATE["config"]
    base_text_cfg = cfg["text_style"]
    text_layout = resolve_text_layout(background_item)
    layout_style = text_layout.get("text_style")
    if isinstance(layout_style, dict):
        text_cfg: dict[str, Any] = deep_merge(dict(base_text_cfg), layout_style)
    else:
        text_cfg = dict(base_text_cfg)
    raw = normalize_mojibake_text(slogan)
    tuning = resolve_text_tuning(raw)
    stroke_width = int(text_cfg.get("stroke_width", 3))
    draw = ImageDraw.Draw(image)
    bounds = resolve_slogan_bounds(image.size, text_layout)
    use_region = bounds is not None

    if slogan_row is None:
        infer_explicit = slogan_explicit_lines(raw)
        slogan_row = len(infer_explicit) if infer_explicit else 1

    if use_region:
        left, reg_top, reg_w, reg_h = bounds
        max_width = reg_w
        max_height = max(reg_h, 20)
        if slogan_row == 3:
            try:
                row3_width_ratio = float(text_layout.get("row3_width_ratio", 0))
            except (TypeError, ValueError):
                row3_width_ratio = 0.0
            if 0.0 < row3_width_ratio <= 1.0:
                row3_w = max(int(image.width * row3_width_ratio), reg_w)
                row3_w = min(row3_w, image.width)
                left = max((image.width - row3_w) // 2, 0)
                reg_w = row3_w
                max_width = reg_w
    else:
        top_m = int(text_cfg.get("top_margin", 88))
        left = 0
        reg_top = top_m
        reg_w = image.width
        max_width = int(image.width * float(tuning["max_width_ratio"]))
        max_height = max(text_layout["top_overlay_height"] - top_m - 12, 20)
        reg_h = max_height

    font_size_min = max(int(text_layout["font_size_min"] * float(tuning["font_scale"])), 1)
    font_size_max = max(int(text_layout["font_size_max"] * float(tuning["font_scale"])), font_size_min)
    if slogan_row == 2:
        try:
            row2_font_scale = float(text_layout.get("row2_font_scale", 1.0))
        except (TypeError, ValueError):
            row2_font_scale = 1.0
        row2_font_scale = min(max(row2_font_scale, 0.6), 1.0)
        font_size_min = max(int(font_size_min * row2_font_scale), 1)
        font_size_max = max(int(font_size_max * row2_font_scale), font_size_min)
    if slogan_row == 3:
        try:
            row3_font_scale = float(text_layout.get("row3_font_scale", 1.0))
        except (TypeError, ValueError):
            row3_font_scale = 1.0
        row3_font_scale = min(max(row3_font_scale, 0.6), 1.0)
        font_size_min = max(int(font_size_min * row3_font_scale), 1)
        font_size_max = max(int(font_size_max * row3_font_scale), font_size_min)
    tuning_max = tuning.get("max_font_size")
    if tuning_max is not None:
        font_size_max = min(font_size_max, max(int(tuning_max), font_size_min))

    explicit = slogan_explicit_lines(raw)
    max_lines_cap = int(text_layout["max_lines"])
    region_row1 = bool(use_region and slogan_row == 1)
    region_row2 = bool(use_region and slogan_row == 2)
    region_row3 = bool(use_region and slogan_row == 3)

    if region_row1:
        single = " ".join(raw.split())
        font, lines, spacing = layout_slogan_lines(
            slogan=single,
            stroke_width=stroke_width,
            max_width=max_width,
            max_height=max_height,
            max_lines=1,
            preferred_lines=1,
            font_size_min=font_size_min,
            font_size_max=font_size_max,
            line_spacing_min=text_layout["line_spacing_min"],
            line_spacing_max=text_layout["line_spacing_max"],
            enable_punctuation_break=False,
            balance_weight=int(tuning["balance_weight"]),
            line_priority=[1],
        )
    elif region_row2:
        if explicit:
            lines_in = normalize_slogan_lines_to_row_count(explicit, 2)
            font, lines, spacing = layout_slogan_lines_fixed(
                lines=lines_in,
                stroke_width=stroke_width,
                max_width=max_width,
                max_height=max_height,
                max_lines=2,
                font_size_min=font_size_min,
                font_size_max=font_size_max,
                line_spacing_min=text_layout["line_spacing_min"],
                line_spacing_max=text_layout["line_spacing_max"],
            )
        else:
            single = " ".join(raw.split())
            font, lines, spacing = layout_slogan_lines(
                slogan=single,
                stroke_width=stroke_width,
                max_width=max_width,
                max_height=max_height,
                max_lines=2,
                preferred_lines=2,
                font_size_min=font_size_min,
                font_size_max=font_size_max,
                line_spacing_min=text_layout["line_spacing_min"],
                line_spacing_max=text_layout["line_spacing_max"],
                enable_punctuation_break=bool(tuning["auto_break_on_punctuation"]),
                balance_weight=int(tuning["balance_weight"]),
                line_priority=tuning.get("line_priority"),
            )
    elif region_row3:
        if explicit:
            lines_in = normalize_slogan_lines_to_row_count(explicit, 3)
            font, lines, spacing = layout_slogan_lines_fixed(
                lines=lines_in,
                stroke_width=stroke_width,
                max_width=max_width,
                max_height=max_height,
                max_lines=3,
                font_size_min=font_size_min,
                font_size_max=font_size_max,
                line_spacing_min=text_layout["line_spacing_min"],
                line_spacing_max=text_layout["line_spacing_max"],
            )
        else:
            single = " ".join(raw.split())
            font, lines, spacing = layout_slogan_lines(
                slogan=single,
                stroke_width=stroke_width,
                max_width=max_width,
                max_height=max_height,
                max_lines=3,
                preferred_lines=3,
                font_size_min=font_size_min,
                font_size_max=font_size_max,
                line_spacing_min=text_layout["line_spacing_min"],
                line_spacing_max=text_layout["line_spacing_max"],
                enable_punctuation_break=bool(tuning["auto_break_on_punctuation"]),
                balance_weight=int(tuning["balance_weight"]),
                line_priority=tuning.get("line_priority"),
            )
    elif explicit:
        lines_for_layout = (
            normalize_slogan_lines_to_row_count(explicit, slogan_row) if use_region else explicit
        )
        font, lines, spacing = layout_slogan_lines_fixed(
            lines=lines_for_layout,
            stroke_width=stroke_width,
            max_width=max_width,
            max_height=max_height,
            max_lines=max_lines_cap,
            font_size_min=font_size_min,
            font_size_max=font_size_max,
            line_spacing_min=text_layout["line_spacing_min"],
            line_spacing_max=text_layout["line_spacing_max"],
        )
    else:
        single = " ".join(raw.split())
        font, lines, spacing = layout_slogan_lines(
            slogan=single,
            stroke_width=stroke_width,
            max_width=max_width,
            max_height=max_height,
            max_lines=text_layout["max_lines"],
            preferred_lines=max(int(tuning["preferred_lines"]), int(text_layout["preferred_lines"])),
            font_size_min=font_size_min,
            font_size_max=font_size_max,
            line_spacing_min=text_layout["line_spacing_min"],
            line_spacing_max=text_layout["line_spacing_max"],
            enable_punctuation_break=bool(tuning["auto_break_on_punctuation"]),
            balance_weight=int(tuning["balance_weight"]),
            line_priority=tuning.get("line_priority"),
        )
    font_size_start = int(getattr(font, "size", font_size_max))
    font, spacing = fit_lines_to_region(
        lines=lines,
        stroke_width=stroke_width,
        max_width=max_width,
        max_height=max_height,
        line_spacing_min=text_layout["line_spacing_min"],
        line_spacing_max=text_layout["line_spacing_max"],
        font_size_start=font_size_start,
        font_size_min=max(1, min(font_size_min, font_size_start)),
    )

    line_metrics: list[tuple[str, int, int]] = []
    for line in lines:
        if draw.textbbox((0, 0), line, font=font, stroke_width=stroke_width)[2] > max_width:
            line = ellipsize_to_width(draw, line, font, max_width, stroke_width)
        bbox = draw.textbbox((0, 0), line, font=font, stroke_width=stroke_width)
        line_metrics.append((line, bbox[2] - bbox[0], bbox[3] - bbox[1]))

    text_block_h = sum(item[2] for item in line_metrics) + max(len(line_metrics) - 1, 0) * spacing
    if use_region:
        current_y = reg_top + max((reg_h - text_block_h) // 2, 0) + int(tuning["y_offset"])
    else:
        overlay_top = int(text_cfg.get("top_margin", 88))
        overlay_bottom = max(text_layout["top_overlay_height"] - 12, overlay_top + 1)
        overlay_h = max(overlay_bottom - overlay_top, 1)
        current_y = overlay_top + max((overlay_h - text_block_h) // 2, 0) + int(tuning["y_offset"])
    for line, text_w, text_h in line_metrics:
        if use_region:
            x = left + max((reg_w - text_w) // 2, 0)
        else:
            x = (image.width - text_w) // 2
        try:
            draw_slogan_line_layered(
                image=image,
                line=line,
                position=(x, current_y),
                font=font,
                stroke_width=stroke_width,
                text_cfg=text_cfg,
            )
        except Exception:
            draw.text(
                (x, current_y),
                line,
                font=font,
                fill=text_cfg["fill"],
                stroke_width=stroke_width,
                stroke_fill=text_cfg["stroke_fill"],
            )
        current_y += text_h + spacing
    return image
