"""在四幅背景图上绘制隐形文本框，输出到当前目录。"""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

items = config["background_set"]["items"]
output_w = config["output"]["width"]   # 1080
output_h = config["output"]["height"]  # 1920

for item in items:
    bg_path = ROOT / item["path"]
    if not bg_path.exists():
        print(f"SKIP: {bg_path} 不存在")
        continue

    img = Image.open(bg_path).convert("RGBA")
    orig_w, orig_h = img.size
    print(f"\n{item['name']} ({item['id']})  原图: {orig_w}x{orig_h}")

    # 如果原图不是 1080x1920，等比缩放（保持比例，取 fitting 方式）
    if (orig_w, orig_h) != (output_w, output_h):
        scale = min(output_w / orig_w, output_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        # 创建 1080x1920 画布贴入
        canvas = Image.new("RGBA", (output_w, output_h), (0, 0, 0, 0))
        paste_x = (output_w - new_w) // 2
        paste_y = (output_h - new_h) // 2
        canvas.paste(img, (paste_x, paste_y))
        img = canvas
        print(f"  缩放至: {new_w}x{new_h}, 画布: {output_w}x{output_h}")

    draw = ImageDraw.Draw(img)

    # 获取 text_region 参数
    tr = item["text_layout"]["text_region"]
    top = int(output_h * tr["margin_top_ratio"])      # 77
    width = int(output_w * tr["width_ratio"])           # 994
    height = int(output_h * tr["height_ratio"])         # 403
    left = (output_w - width) // 2                      # 43

    # 绘制半透明黄色填充
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(
        [left, top, left + width, top + height],
        fill=(255, 220, 50, 70),
        outline=(255, 200, 40, 220),
        width=3,
    )
    img = Image.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(img)

    # 标注尺寸信息
    info_lines = [
        f"{item['name']} ({item['id']})",
        f"text_region: {width}x{height}px",
        f"位置: top={top}px, left={left}px",
        f"margin_top_ratio={tr['margin_top_ratio']}",
        f"width_ratio={tr['width_ratio']}",
        f"height_ratio={tr['height_ratio']}",
    ]

    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 28)
        font_small = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 22)
    except OSError:
        font = ImageFont.load_default()
        font_small = font

    info_y = top + height + 20
    for line in info_lines:
        bbox = draw.textbbox((0, 0), line, font=font_small)
        draw.text((left, info_y), line, font=font_small, fill=(255, 220, 50))
        info_y += bbox[3] - bbox[1] + 6

    # 四角标注像素坐标
    corners = [
        (left, top, f"({left},{top})"),
        (left + width, top, f"({left+width},{top})"),
        (left, top + height, f"({left},{top+height})"),
        (left + width, top + height, f"({left+width},{top+height})"),
    ]
    for cx, cy, label in corners:
        draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(255, 80, 60))
        draw.text((cx + 8, cy - 14), label, font=font_small, fill=(255, 255, 255),
                  stroke_width=3, stroke_fill=(0, 0, 0))

    # 中心十字
    cx, cy = left + width // 2, top + height // 2
    draw.line([(cx - 20, cy), (cx + 20, cy)], fill=(255, 255, 255, 180), width=2)
    draw.line([(cx, cy - 20), (cx, cy + 20)], fill=(255, 255, 255, 180), width=2)

    out_path = Path(__file__).parent / f"{item['id']}_text_region.png"
    img_rgb = Image.new("RGB", img.size, (30, 20, 10))
    img_rgb.paste(img, mask=img.split()[3])
    img_rgb.save(out_path, quality=90)
    print(f"  已保存: {out_path.name}")

print("\n完成！请查看 text_region_preview/ 目录中的 4 张预览图。")
