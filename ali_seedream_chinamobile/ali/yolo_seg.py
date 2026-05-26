from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Union

import cv2
import numpy as np

if TYPE_CHECKING:
    from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = ROOT / "output"


class YoloLargestSegmentMask:
    """YOLO 分割：取面积最大实例，输出与输入同尺寸的透明底图（cv2 BGRA）。"""

    def __init__(
        self,
        model_path: str = "yolo26x-seg.pt",
        output_dir: Path | str | None = None,
    ) -> None:
        from ultralytics import YOLO  # 懒加载，避免非 YOLO 管线触发 torch 依赖

        self.model = YOLO(model_path)
        self.output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR

    @staticmethod
    def _to_bgr(image: Union["Image.Image", np.ndarray]) -> np.ndarray:
        """统一为 BGR uint8 ndarray，供 Ultralytics 使用。"""
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            if image.ndim == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            if image.shape[2] == 3:
                return np.ascontiguousarray(image)
            raise ValueError(f"不支持的 ndarray 形状: {image.shape}")

        from PIL import Image as PILImage

        if not isinstance(image, PILImage.Image):
            raise TypeError("image 须为 PIL.Image.Image 或 numpy.ndarray")

        if image.mode in ("RGBA", "LA"):
            rgba = np.array(image.convert("RGBA"))
            return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        rgb = np.array(image.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def process(
        self,
        image: Union["Image.Image", np.ndarray],
        *,
        save: bool = True,
        save_path: Path | str | None = None,
        save_stem: str = "largest_seg",
    ) -> np.ndarray:
        """
        Args:
            image: PIL 图片，或 cv2 用 BGR/BGRA/灰度 ndarray (uint8)。
            save: 是否写入磁盘。
            save_path: 完整输出路径；若指定则忽略 save_stem。
            save_stem: 未给 save_path 时，保存为 ``{output_dir}/{save_stem}_largest_seg_rgba.png``。

        Returns:
            与输入同尺寸的 BGRA ndarray（cv2 四通道惯例），背景 alpha=0。
        """
        bgr = self._to_bgr(image)
        h, w = bgr.shape[:2]

        results = self.model.predict(
            source=bgr,
            retina_masks=True,
            save=False,
            show=False,
            verbose=False,
        )
        r = results[0]
        if r.masks is None or r.masks.data.numel() == 0:
            raise RuntimeError("未得到分割掩码，请确认输入图与模型。")

        masks = r.masks.data.cpu().numpy()
        areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
        idx = int(np.argmax(areas))
        m = masks[idx]
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

        mask_bool = m > 0.5

        bgra = np.zeros((h, w, 4), dtype=np.uint8)
        bgra[mask_bool, 0] = bgr[mask_bool, 0]
        bgra[mask_bool, 1] = bgr[mask_bool, 1]
        bgra[mask_bool, 2] = bgr[mask_bool, 2]
        bgra[mask_bool, 3] = 255

        if save:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if save_path is not None:
                out = Path(save_path)
            else:
                out = self.output_dir / f"{save_stem}_largest_seg_rgba.png"
            ok = cv2.imwrite(str(out), bgra)
            if not ok:
                raise RuntimeError(f"无法写入: {out}")
            print(f"Saved: {out}")

        return bgra


if __name__ == "__main__":
    INPUT_PATH = ROOT / "resource" / "person_front" / "重叠人像_1500x2000.jpg"
    bgr_in = cv2.imread(str(INPUT_PATH))
    if bgr_in is None:
        raise FileNotFoundError(INPUT_PATH)
    extractor = YoloLargestSegmentMask()
    _ = extractor.process(bgr_in, save_stem=INPUT_PATH.stem)
