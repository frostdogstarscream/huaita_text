"""批量渲染：4 背景 × 75 标语，输出到 1/2/3/4 子文件夹。"""
import json, sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app_state import get_app_paths, APP_STATE
get_app_paths()

from config_manager import load_config
config = load_config()
APP_STATE["config"] = config

from text_renderer import draw_slogan

OUT = Path(__file__).parent

bg_items = config["background_set"]["items"]
slogans = config["rotation"]["slogans"]

for bg_idx, bg_item in enumerate(bg_items):
    bg_id = bg_item["id"]  # bg_001 ~ bg_004
    folder_num = str(bg_idx + 1)
    out_dir = OUT / folder_num
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载背景并缩放至 1080×1920
    bg_path = ROOT / bg_item["path"]
    if not bg_path.exists():
        print(f"SKIP: {bg_path} 不存在")
        continue

    bg_img = Image.open(bg_path).convert("RGBA")
    orig_w, orig_h = bg_img.size
    if (orig_w, orig_h) != (1080, 1920):
        scale = min(1080 / orig_w, 1920 / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        bg_img = bg_img.resize((new_w, new_h), Image.LANCZOS)
        canvas = Image.new("RGBA", (1080, 1920), (0, 0, 0, 0))
        canvas.paste(bg_img, ((1080 - new_w) // 2, (1920 - new_h) // 2))
        bg_img = canvas

    print(f"\n{bg_item['name']} ({bg_id}) -> {folder_num}/")

    for i, s in enumerate(slogans):
        content = s["content"]
        row = s["row"]

        # 每次用背景副本渲染
        img = bg_img.copy()
        try:
            result = draw_slogan(img, content, bg_item, slogan_row=row)
        except Exception as exc:
            print(f"  [{i+1:02d}] ERROR: {content[:20]}...  {exc}")
            continue

        # 转 RGB 保存 JPEG
        result_rgb = Image.new("RGB", result.size, (30, 20, 10))
        result_rgb.paste(result, mask=result.split()[3])

        safe_name = f"{i+1:02d}_{content[:12].replace(' ', '_').replace(chr(10), '_')}"
        out_path = out_dir / f"{safe_name}.jpg"
        result_rgb.save(out_path, quality=85)

        if (i + 1) % 15 == 0:
            print(f"  [{i+1:02d}/75] ...")

    print(f"  完成: {len(slogans)} 张 -> {out_dir}")

print("\n全部完成！")
